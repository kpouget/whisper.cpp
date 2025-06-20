#include <stdio.h>
#include <cassert>
#include <cerrno>
#include <unistd.h>

#include <cstdlib>

#include "virtgpu.h"

static virt_gpu_result_t virtgpu_open_device(struct virtgpu *gpu, const drmDevicePtr dev);
static virt_gpu_result_t virtgpu_open(struct virtgpu *gpu);


static virt_gpu_result_t virtgpu_init_params(struct virtgpu *gpu);
static virt_gpu_result_t virtgpu_init_capset(struct virtgpu *gpu);
static virt_gpu_result_t virtgpu_init_context(struct virtgpu *gpu);

static int virtgpu_ioctl_context_init(struct virtgpu *gpu,
				      enum virgl_renderer_capset capset_id);
static int
virtgpu_ioctl_get_caps(struct virtgpu *gpu,
                       enum virgl_renderer_capset id,
                       uint32_t version,
                       void *capset,
                       size_t capset_size);
static uint64_t virtgpu_ioctl_getparam(struct virtgpu *gpu, uint64_t param);
static void virtgpu_init_renderer_info(struct virtgpu *gpu);

static inline void
virtgpu_init_shmem_blob_mem(struct virtgpu *gpu)
{
   /* VIRTGPU_BLOB_MEM_GUEST allocates from the guest system memory.  They are
    * logically contiguous in the guest but are sglists (iovecs) in the host.
    * That makes them slower to process in the host.  With host process
    * isolation, it also becomes impossible for the host to access sglists
    * directly.
    *
    * While there are ideas (and shipped code in some cases) such as creating
    * udmabufs from sglists, or having a dedicated guest heap, it seems the
    * easiest way is to reuse VIRTGPU_BLOB_MEM_HOST3D.  That is, when the
    * renderer sees a request to export a blob where
    *
    *  - blob_mem is VIRTGPU_BLOB_MEM_HOST3D
    *  - blob_flags is VIRTGPU_BLOB_FLAG_USE_MAPPABLE
    *  - blob_id is 0
    *
    * it allocates a host shmem.
    *
    * supports_blob_id_0 has been enforced by mandated render server config.
    */
   assert(gpu->capset.data.supports_blob_id_0);
   gpu->shmem_blob_mem = VIRTGPU_BLOB_MEM_HOST3D;
}

struct virtgpu *
create_virtgpu() {
  struct virtgpu *gpu = new struct virtgpu();

  util_sparse_array_init(&gpu->shmem_array, sizeof(struct virtgpu_shmem),
			 1024);

  virt_gpu_result_t result = virtgpu_open(gpu);
  assert(result == APIR_SUCCESS);

  result = virtgpu_init_params(gpu);
  assert(result == APIR_SUCCESS);

  result = virtgpu_init_capset(gpu);
  assert(result == APIR_SUCCESS);

  result = virtgpu_init_context(gpu);
  assert(result == APIR_SUCCESS);

#ifdef NDEBUG
   UNUSED(result);
#endif

  virtgpu_init_shmem_blob_mem(gpu);

  gpu->reply_shmem = virtgpu_shmem_create(gpu, 0x4000);
  gpu->data_shmem = virtgpu_shmem_create(gpu, 0x1830000); // 24MiB

  if (!gpu->reply_shmem) {
    FATAL("%s: failed to create the shared reply memory pages :/", __func__);
  }

  if (!gpu->data_shmem) {
    FATAL("%s: failed to create the shared data memory pages :/", __func__);
  }

  struct vn_cs_encoder *encoder;
  struct vn_cs_decoder *decoder;
  int32_t ret;

  encoder = remote_call_prepare(gpu,  VIRGL_APIR_COMMAND_TYPE_LoadLibrary, 0);
  if (!encoder) {
    FATAL("%s: failed to prepare the remote call encoder :/", __func__);
  }
  decoder = remote_call(gpu, encoder);
  if (!decoder) {
    FATAL("%s: failed to kick the remote call :/", __func__);
  }

  ret = remote_call_finish(encoder, decoder);
  if (ret != 0) {
    FATAL("%s: failed to load the APIR backend libraries (code=%d):/", __func__, ret);
  }

  return gpu;
}


static virt_gpu_result_t
virtgpu_open(struct virtgpu *gpu)
{
   drmDevicePtr devs[8];
   int count = drmGetDevices2(0, devs, ARRAY_SIZE(devs));
   if (count < 0) {
     INFO("failed to enumerate DRM devices");
     return APIR_ERROR_INITIALIZATION_FAILED;
   }

   virt_gpu_result_t result = APIR_ERROR_INITIALIZATION_FAILED;
   for (int i = 0; i < count; i++) {
      result = virtgpu_open_device(gpu, devs[i]);
      if (result == APIR_SUCCESS)
         break;
   }

   drmFreeDevices(devs, count);

   return result;
}

static virt_gpu_result_t
virtgpu_open_device(struct virtgpu *gpu, const drmDevicePtr dev)
{
   bool supported_bus = false;

   switch (dev->bustype) {
   case DRM_BUS_PCI:
      if (dev->deviceinfo.pci->vendor_id == VIRTGPU_PCI_VENDOR_ID &&
          dev->deviceinfo.pci->device_id == VIRTGPU_PCI_DEVICE_ID)
         supported_bus = true;
      break;
   case DRM_BUS_PLATFORM:
      supported_bus = true;
      break;
   default:
      break;
   }

   if (!supported_bus || !(dev->available_nodes & (1 << DRM_NODE_RENDER))) {
      if (VN_DEBUG(INIT)) {
         const char *name = "unknown";
         for (uint32_t i = 0; i < DRM_NODE_MAX; i++) {
            if (dev->available_nodes & (1 << i)) {
               name = dev->nodes[i];
               break;
            }
         }
         vn_log(gpu->instance, "skipping DRM device %s", name);
      }
      return APIR_ERROR_INITIALIZATION_FAILED;
   }

   const char *primary_path = dev->nodes[DRM_NODE_PRIMARY];
   const char *node_path = dev->nodes[DRM_NODE_RENDER];

   int fd = open(node_path, O_RDWR | O_CLOEXEC);
   if (fd < 0) {
      if (VN_DEBUG(INIT))
         vn_log(gpu->instance, "failed to open %s", node_path);
      return APIR_ERROR_INITIALIZATION_FAILED;
   }

   drmVersionPtr version = drmGetVersion(fd);
   if (!version || strcmp(version->name, "virtio_gpu") ||
       version->version_major != 0) {
      if (VN_DEBUG(INIT)) {
         if (version) {
            vn_log(gpu->instance, "unknown DRM driver %s version %d",
                   version->name, version->version_major);
         } else {
            vn_log(gpu->instance, "failed to get DRM driver version");
         }
      }
      if (version)
         drmFreeVersion(version);
      close(fd);
      return APIR_ERROR_INITIALIZATION_FAILED;
   }

   gpu->fd = fd;

   struct stat st;
   if (stat(primary_path, &st) == 0) {
      gpu->has_primary = true;
      gpu->primary_major = major(st.st_rdev);
      gpu->primary_minor = minor(st.st_rdev);
   } else {
      gpu->has_primary = false;
      gpu->primary_major = 0;
      gpu->primary_minor = 0;
   }
   stat(node_path, &st);
   gpu->render_major = major(st.st_rdev);
   gpu->render_minor = minor(st.st_rdev);

   gpu->bustype = dev->bustype;
   if (dev->bustype == DRM_BUS_PCI)
      gpu->pci_bus_info = *dev->businfo.pci;

   drmFreeVersion(version);

   INFO("using DRM device %s", node_path);

   return APIR_SUCCESS;
}

void
vn_log(struct remoting_dev_instance *instance, const char *format, ...)
{
   if (instance) {
     printf("<INST>");
   }

   va_list ap;

   va_start(ap, format);
   vprintf(format, ap);
   va_end(ap);

   /* instance may be NULL or partially initialized */
}

static virt_gpu_result_t
virtgpu_init_context(struct virtgpu *gpu)
{
   assert(!gpu->capset.version);
   const int ret = virtgpu_ioctl_context_init(gpu, gpu->capset.id);
   if (ret) {
      if (VN_DEBUG(INIT)) {
         vn_log(gpu->instance, "failed to initialize context: %s",
                strerror(errno));
      }
      return APIR_ERROR_INITIALIZATION_FAILED;
   }

   return APIR_SUCCESS;
}

static virt_gpu_result_t
virtgpu_init_capset(struct virtgpu *gpu)
{
   gpu->capset.id = VIRGL_RENDERER_CAPSET_VENUS;
   gpu->capset.version = 0;

   const int ret =
      virtgpu_ioctl_get_caps(gpu, gpu->capset.id, gpu->capset.version,
                             &gpu->capset.data, sizeof(gpu->capset.data));
   if (ret) {
      if (VN_DEBUG(INIT)) {
         vn_log(gpu->instance, "failed to get venus v%d capset: %s",
                gpu->capset.version, strerror(errno));
      }
      return APIR_ERROR_INITIALIZATION_FAILED;
   }

   return APIR_SUCCESS;
}

static virt_gpu_result_t
virtgpu_init_params(struct virtgpu *gpu)
{
   const uint64_t required_params[] = {
      VIRTGPU_PARAM_3D_FEATURES,   VIRTGPU_PARAM_CAPSET_QUERY_FIX,
      VIRTGPU_PARAM_RESOURCE_BLOB, VIRTGPU_PARAM_CONTEXT_INIT,
   };
   uint64_t val;
   for (uint32_t i = 0; i < ARRAY_SIZE(required_params); i++) {
      val = virtgpu_ioctl_getparam(gpu, required_params[i]);
      if (!val) {
         if (VN_DEBUG(INIT)) {
            vn_log(gpu->instance, "required kernel param %d is missing",
                   (int)required_params[i]);
         }
         return APIR_ERROR_INITIALIZATION_FAILED;
      }
   }

   val = virtgpu_ioctl_getparam(gpu, VIRTGPU_PARAM_HOST_VISIBLE);
   if (val) {
      gpu->bo_blob_mem = VIRTGPU_BLOB_MEM_HOST3D;
   } else {
      val = virtgpu_ioctl_getparam(gpu, VIRTGPU_PARAM_GUEST_VRAM);
      if (val) {
         gpu->bo_blob_mem = VIRTGPU_BLOB_MEM_GUEST_VRAM;
      }
   }

   if (!val) {
      vn_log(gpu->instance,
             "one of required kernel params (%d or %d) is missing",
             (int)VIRTGPU_PARAM_HOST_VISIBLE, (int)VIRTGPU_PARAM_GUEST_VRAM);
      return APIR_ERROR_INITIALIZATION_FAILED;
   }

   /* Cross-device feature is optional.  It enables sharing dma-bufs
    * with other virtio devices, like virtio-wl or virtio-video used
    * by ChromeOS VMs.  Qemu doesn't support cross-device sharing.
    */
   val = virtgpu_ioctl_getparam(gpu, VIRTGPU_PARAM_CROSS_DEVICE);
   if (val)
      gpu->supports_cross_device = true;

   /* implied by CONTEXT_INIT uapi */
   gpu->max_timeline_count = 64;

   return APIR_SUCCESS;
}

static int
virtgpu_ioctl_context_init(struct virtgpu *gpu,
                           enum virgl_renderer_capset capset_id)
{
   struct drm_virtgpu_context_set_param ctx_set_params[3] = {
      {
         .param = VIRTGPU_CONTEXT_PARAM_CAPSET_ID,
         .value = capset_id,
      },
      {
         .param = VIRTGPU_CONTEXT_PARAM_NUM_RINGS,
         .value = 64,
      },
      {
         .param = VIRTGPU_CONTEXT_PARAM_POLL_RINGS_MASK,
         .value = 0, /* don't generate drm_events on fence signaling */
      },
   };

   struct drm_virtgpu_context_init args = {
      .num_params = ARRAY_SIZE(ctx_set_params),
      .pad = 0,
      .ctx_set_params = (uintptr_t)&ctx_set_params,
   };

   return virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_CONTEXT_INIT, &args);
}

static int
virtgpu_ioctl_get_caps(struct virtgpu *gpu,
                       enum virgl_renderer_capset id,
                       uint32_t version,
                       void *capset,
                       size_t capset_size)
{
   struct drm_virtgpu_get_caps args = {
      .cap_set_id = id,
      .cap_set_ver = version,
      .addr = (uintptr_t)capset,
      .size = (__u32) capset_size,
      .pad = 0,
   };

   return virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_GET_CAPS, &args);
}

static uint64_t
virtgpu_ioctl_getparam(struct virtgpu *gpu, uint64_t param)
{
   /* val must be zeroed because kernel only writes the lower 32 bits */
   uint64_t val = 0;
   struct drm_virtgpu_getparam args = {
      .param = param,
      .value = (uintptr_t)&val,
   };

   const int ret = virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_GETPARAM, &args);
   return ret ? 0 : val;
}


struct vn_cs_encoder *
remote_call_prepare(
  struct virtgpu *gpu,
  int32_t cmd_type,
  int32_t cmd_flags)
{

  if (!gpu->reply_shmem) {
    FATAL("%s: the reply shmem page can't be null", __func__);
  }

  /*
   * Prepare the command encoder and its buffer
   */

  static char encoder_buffer[4096];

  static struct vn_cs_encoder enc;
  enc = {
    encoder_buffer,
    encoder_buffer,
    encoder_buffer + sizeof(encoder_buffer),
  };

  /*
   * Fill the command encoder with the common args:
   * - cmd_type (int32_t)
   * - cmd_flags (int32_t)
   * - reply res id (uint32_t)
   */

  vn_encode_int32_t(&enc, &cmd_type);
  vn_encode_int32_t(&enc, &cmd_flags);

  uint32_t reply_res_id = gpu->reply_shmem->res_id;
  vn_encode_uint32_t(&enc, &reply_res_id);

  return &enc;
}

int32_t
remote_call_finish(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec) {
  if (!enc) {
    WARNING("Invalid (null) encoder :/");
  }
  if (!dec) {
    FATAL("Invalid (null) decoder :/");
  }
  int32_t remote_call_ret;
  vn_decode_int32_t(dec, &remote_call_ret);

  // encoder and decoder are statically allocated, nothing to do to release them

  return remote_call_ret;
}

struct vn_cs_decoder *
remote_call(
  struct virtgpu *gpu,
  struct vn_cs_encoder *encoder
  )
{
  /*
   * Prepare the reply notification pointer
   */

  volatile std::atomic_uint *atomic_reply_notif = (volatile std::atomic_uint *) gpu->reply_shmem->mmap_ptr;
  *atomic_reply_notif = 0;

  /*
   * Trigger the execbuf ioctl
   */

  struct drm_virtgpu_execbuffer args = {
    .flags = VIRTGPU_EXECBUF_RING_IDX,
    .size = (uint32_t) (encoder->cur - encoder->start),
    .command = (uintptr_t) encoder->start,

    .bo_handles = 0,
    .num_bo_handles = 0,

    .fence_fd = 0,
    .ring_idx = 0,
    .syncobj_stride = 0,
    .num_in_syncobjs = 0,
    .num_out_syncobjs = 0,
    .in_syncobjs = 0,
    .out_syncobjs = 0,
  };

  int ret = drmIoctl(gpu->fd, DRM_IOCTL_VIRTGPU_EXECBUFFER, &args);

  if (ret != 0) {
    FATAL("%s: the virtgpu EXECBUFFER ioctl failed (%d) :/ \n", ret);
  }
  /*
   * Wait for the response notification
   */

  while (std::atomic_load_explicit(atomic_reply_notif, std::memory_order_acquire) == 0) {
    int64_t base_sleep_us = 15;

    os_time_sleep(base_sleep_us);
  }

  /*
   * Prepare the decoder
   */
  static struct vn_cs_decoder dec;
  dec.cur = (char *) gpu->reply_shmem->mmap_ptr + sizeof(*atomic_reply_notif);
  dec.end = (char *) gpu->reply_shmem->mmap_ptr + gpu->reply_shmem->mmap_size;

  return &dec;
}
