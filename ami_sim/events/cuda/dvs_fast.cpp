#include <torch/extension.h>

#include <tuple>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> generate_events_cuda(
    torch::Tensor mu,
    torch::Tensor sigma,
    torch::Tensor residual,
    double start_t,
    double end_t,
    int64_t width,
    int64_t height,
    int64_t seed,
    int64_t frame_index,
    int64_t max_events_per_pixel);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> generate_events(
    torch::Tensor mu,
    torch::Tensor sigma,
    torch::Tensor residual,
    double start_t,
    double end_t,
    int64_t width,
    int64_t height,
    int64_t seed,
    int64_t frame_index,
    int64_t max_events_per_pixel) {
  TORCH_CHECK(mu.is_cuda(), "dvs_fast requires CUDA tensors");
  TORCH_CHECK(sigma.is_cuda() && residual.is_cuda(), "all state tensors must be CUDA tensors");
  TORCH_CHECK(mu.is_contiguous() && sigma.is_contiguous() && residual.is_contiguous(),
              "state tensors must be contiguous");
  TORCH_CHECK(mu.dim() == 2, "mu must have shape [H,W]");
  TORCH_CHECK(mu.sizes() == sigma.sizes() && mu.sizes() == residual.sizes(),
              "state tensors must have matching shapes");
  TORCH_CHECK(mu.scalar_type() == sigma.scalar_type() && mu.scalar_type() == residual.scalar_type(),
              "state tensors must have matching dtypes");
  TORCH_CHECK(width == mu.size(1) && height == mu.size(0), "width/height do not match state tensors");
  TORCH_CHECK(max_events_per_pixel > 0, "max_events_per_pixel must be positive");
  TORCH_CHECK(mu.scalar_type() == torch::kFloat || mu.scalar_type() == torch::kDouble,
              "dvs_fast supports float32 and float64 state tensors");
  return generate_events_cuda(mu, sigma, residual, start_t, end_t, width, height,
                              seed, frame_index, max_events_per_pixel);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("generate_events", &generate_events, "Generate DVS events (CUDA)");
}
