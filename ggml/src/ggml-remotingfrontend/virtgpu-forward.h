#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-alloc.h"

#include "virtgpu-utils.h"

#include "../ggml-remotingbackend/shared/apir_backend.h"

/* device */
int apir_device_get_count(struct virtgpu *gpu);
const char *apir_device_get_name(struct virtgpu *gpu);
const char *apir_device_get_description(struct virtgpu *gpu);
uint32_t apir_device_get_type(struct virtgpu *gpu);
void apir_device_get_memory(struct virtgpu *gpu, size_t *free, size_t *total);
bool apir_device_supports_op(struct virtgpu *gpu, const ggml_tensor *op);
apir_buffer_type_host_handle_t apir_device_get_buffer_type(struct virtgpu *gpu);
void apir_device_get_props(struct virtgpu *gpu,
			   bool *async,
			   bool *host_buffer,
			   bool *buffer_from_host_ptr,
			   bool *events);
apir_buffer_context_t apir_device_buffer_from_ptr(struct virtgpu *gpu,
						  size_t size,
						  size_t max_tensor_size);
/* buffer-type */
const char *apir_buffer_type_get_name(struct virtgpu *gpu, ggml_backend_buffer_type_t buft);
size_t apir_buffer_type_get_alignment(struct virtgpu *gpu, ggml_backend_buffer_type_t buft);
size_t apir_buffer_type_get_max_size(struct virtgpu *gpu, ggml_backend_buffer_type_t buft);
bool apir_buffer_type_is_host(struct virtgpu *gpu, ggml_backend_buffer_type_t buft);
apir_buffer_context_t apir_buffer_type_alloc_buffer(struct virtgpu *gpu, ggml_backend_buffer_type_t buffer_buft, size_t size);

/* buffer */

void *apir_buffer_get_base(struct virtgpu *gpu, apir_buffer_context_t *buffer_context);
enum ggml_status apir_buffer_init_tensor(struct virtgpu *gpu, apir_buffer_context_t *buffer_context, ggml_tensor *tensor);
void apir_buffer_set_tensor(struct virtgpu *gpu, apir_buffer_context_t *buffer_context,
			    ggml_tensor *tensor, const void *data, size_t offset, size_t size);
void apir_buffer_get_tensor(struct virtgpu *gpu, apir_buffer_context_t *buffer_context,
			    const ggml_tensor *tensor, void *data, size_t offset, size_t size);
void apir_buffer_clear(struct virtgpu *gpu, apir_buffer_context_t *buffer_context,
		       uint8_t value);
void apir_buffer_free_buffer(struct virtgpu *gpu, apir_buffer_context_t *buffer_context);

/* backend */

ggml_status apir_backend_graph_compute(struct virtgpu *gpu, ggml_cgraph *cgraph);
