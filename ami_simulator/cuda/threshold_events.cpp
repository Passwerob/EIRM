#include <torch/extension.h>

#include <tuple>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
generate_threshold_events_cuda(
    torch::Tensor log_before,
    torch::Tensor log_after,
    torch::Tensor reference,
    double start_t,
    double end_t,
    double contrast_threshold,
    int64_t max_events_per_pixel);

std::tuple<torch::Tensor, torch::Tensor>
generate_threshold_event_sequence_cuda(
    torch::Tensor log_frames,
    torch::Tensor timestamps,
    double contrast_threshold,
    int64_t max_events_per_pixel,
    double time_origin);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
generate_threshold_events(
    torch::Tensor log_before,
    torch::Tensor log_after,
    torch::Tensor reference,
    double start_t,
    double end_t,
    double contrast_threshold,
    int64_t max_events_per_pixel) {
  TORCH_CHECK(log_before.is_cuda() && log_after.is_cuda() && reference.is_cuda(),
              "compiled threshold backend requires CUDA tensors");
  TORCH_CHECK(log_before.is_contiguous() && log_after.is_contiguous() && reference.is_contiguous(),
              "compiled threshold backend requires contiguous tensors");
  TORCH_CHECK(log_before.dim() == 2 && log_before.sizes() == log_after.sizes() &&
                  log_before.sizes() == reference.sizes(),
              "all state tensors must have matching [H,W] shapes");
  TORCH_CHECK(log_before.scalar_type() == torch::kFloat && log_after.scalar_type() == torch::kFloat &&
                  reference.scalar_type() == torch::kFloat,
              "compiled threshold backend currently uses float32 log-intensity tensors");
  TORCH_CHECK(end_t > start_t, "interval end timestamp must be greater than the start timestamp");
  TORCH_CHECK(contrast_threshold > 0.0, "contrast threshold must be positive");
  TORCH_CHECK(max_events_per_pixel > 0, "max_events_per_pixel must be positive");
  return generate_threshold_events_cuda(
      log_before, log_after, reference, start_t, end_t, contrast_threshold, max_events_per_pixel);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("generate_threshold_events", &generate_threshold_events,
        "Generate one deterministic threshold-event interval (CUDA)");
  m.def("generate_threshold_event_sequence", &generate_threshold_event_sequence_cuda,
        "Generate a deterministic threshold-event sequence with two CUDA passes");
}
