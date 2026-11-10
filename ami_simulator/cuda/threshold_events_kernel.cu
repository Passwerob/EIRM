#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>
#include <limits>
#include <tuple>

namespace {

__global__ void threshold_events_kernel(
    const float* __restrict__ log_before,
    const float* __restrict__ log_after,
    const float* __restrict__ reference,
    float* __restrict__ reference_out,
    int32_t* __restrict__ events,
    double* __restrict__ event_times,
    unsigned long long* __restrict__ event_count,
    uint8_t* __restrict__ overflow,
    int64_t pixels,
    int64_t width,
    double start_t,
    double end_t,
    float threshold,
    int64_t max_events_per_pixel) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pixels) return;

  const float before = log_before[index];
  const float after = log_after[index];
  const float reference_value = reference[index];
  const float delta = after - reference_value;
  const int positive_count = static_cast<int>(floorf(fmaxf(delta, 0.0f) / threshold + 1e-6f));
  const int negative_count = static_cast<int>(floorf(fmaxf(-delta, 0.0f) / threshold + 1e-6f));
  const int total = positive_count + negative_count;
  reference_out[index] = reference_value + static_cast<float>(positive_count - negative_count) * threshold;
  if (total == 0) return;
  if (total > max_events_per_pixel) {
    *overflow = 1;
    return;
  }

  const int64_t slot = static_cast<int64_t>(atomicAdd(event_count, static_cast<unsigned long long>(total)));
  const float slope = after - before;
  const int32_t u = static_cast<int32_t>(index % width);
  const int32_t v = static_cast<int32_t>(index / width);
  int emitted = 0;
  for (int rank = 1; rank <= positive_count; ++rank, ++emitted) {
    const float level = reference_value + threshold * static_cast<float>(rank);
    const float alpha = fabsf(slope) > 1e-6f ? fminf(fmaxf((level - before) / slope, 0.0f), 1.0f) : 0.5f;
    events[(slot + emitted) * 3 + 0] = u;
    events[(slot + emitted) * 3 + 1] = v;
    events[(slot + emitted) * 3 + 2] = 1;
    event_times[slot + emitted] = start_t + static_cast<double>(alpha) * (end_t - start_t);
  }
  for (int rank = 1; rank <= negative_count; ++rank, ++emitted) {
    const float level = reference_value - threshold * static_cast<float>(rank);
    const float alpha = fabsf(slope) > 1e-6f ? fminf(fmaxf((level - before) / slope, 0.0f), 1.0f) : 0.5f;
    events[(slot + emitted) * 3 + 0] = u;
    events[(slot + emitted) * 3 + 1] = v;
    events[(slot + emitted) * 3 + 2] = -1;
    event_times[slot + emitted] = start_t + static_cast<double>(alpha) * (end_t - start_t);
  }
}

__global__ void count_sequence_kernel(
    const float* __restrict__ log_frames,
    float* __restrict__ references,
    int32_t* __restrict__ counts,
    uint8_t* __restrict__ overflow,
    int64_t pixels,
    int64_t intervals,
    float threshold,
    int64_t max_events_per_pixel) {
  const int64_t pixel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pixel >= pixels) return;
  float reference = log_frames[pixel];
  for (int64_t interval = 0; interval < intervals; ++interval) {
    const int64_t offset = interval * pixels + pixel;
    const float before = log_frames[offset];
    const float after = log_frames[offset + pixels];
    const float delta = after - reference;
    const int positive_count = static_cast<int>(floorf(fmaxf(delta, 0.0f) / threshold + 1e-6f));
    const int negative_count = static_cast<int>(floorf(fmaxf(-delta, 0.0f) / threshold + 1e-6f));
    const int total = positive_count + negative_count;
    references[offset] = reference;
    counts[offset] = total;
    if (total > max_events_per_pixel) *overflow = 1;
    reference += static_cast<float>(positive_count - negative_count) * threshold;
  }
}

__global__ void write_sequence_kernel(
    const float* __restrict__ log_frames,
    const float* __restrict__ references,
    const int32_t* __restrict__ counts,
    const int64_t* __restrict__ offsets,
    const double* __restrict__ timestamps,
    int32_t* __restrict__ events,
    int64_t pixels,
    int64_t width,
    int64_t intervals,
    float threshold,
    double time_origin) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_items = intervals * pixels;
  if (index >= total_items) return;
  const int count = counts[index];
  if (count == 0) return;
  const int64_t interval = index / pixels;
  const int64_t pixel = index - interval * pixels;
  const float before = log_frames[index];
  const float after = log_frames[index + pixels];
  const float reference = references[index];
  const float slope = after - before;
  const double start_t = timestamps[interval];
  const double end_t = timestamps[interval + 1];
  const int32_t u = static_cast<int32_t>(pixel % width);
  const int32_t v = static_cast<int32_t>(pixel / width);
  const int positive_count = after >= reference
      ? static_cast<int>(floorf((after - reference) / threshold + 1e-6f))
      : 0;
  int emitted = 0;
  for (int rank = 1; rank <= positive_count; ++rank, ++emitted) {
    const float level = reference + threshold * static_cast<float>(rank);
    const float alpha = fabsf(slope) > 1e-6f ? fminf(fmaxf((level - before) / slope, 0.0f), 1.0f) : 0.5f;
    const int64_t slot = offsets[index] + emitted;
    events[slot * 4 + 0] = static_cast<int32_t>(llrint((start_t + static_cast<double>(alpha) * (end_t - start_t) - time_origin) * 1e6));
    events[slot * 4 + 1] = u;
    events[slot * 4 + 2] = v;
    events[slot * 4 + 3] = 1;
  }
  const int negative_count = count - positive_count;
  for (int rank = 1; rank <= negative_count; ++rank, ++emitted) {
    const float level = reference - threshold * static_cast<float>(rank);
    const float alpha = fabsf(slope) > 1e-6f ? fminf(fmaxf((level - before) / slope, 0.0f), 1.0f) : 0.5f;
    const int64_t slot = offsets[index] + emitted;
    events[slot * 4 + 0] = static_cast<int32_t>(llrint((start_t + static_cast<double>(alpha) * (end_t - start_t) - time_origin) * 1e6));
    events[slot * 4 + 1] = u;
    events[slot * 4 + 2] = v;
    events[slot * 4 + 3] = -1;
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
generate_threshold_events_cuda(
    torch::Tensor log_before,
    torch::Tensor log_after,
    torch::Tensor reference,
    double start_t,
    double end_t,
    double contrast_threshold,
    int64_t max_events_per_pixel) {
  const int64_t height = log_before.size(0);
  const int64_t width = log_before.size(1);
  const int64_t pixels = height * width;
  TORCH_CHECK(pixels > 0, "empty frames are not supported");
  TORCH_CHECK(max_events_per_pixel <= std::numeric_limits<int64_t>::max() / pixels,
              "max_events_per_pixel is too large for this resolution");
  const int64_t capacity = pixels * max_events_per_pixel;
  const auto device = log_before.device();
  const auto event_options = torch::TensorOptions().device(device).dtype(torch::kInt32);
  const auto time_options = torch::TensorOptions().device(device).dtype(torch::kFloat64);
  const auto events = torch::empty({capacity, 3}, event_options);
  const auto event_times = torch::empty({capacity}, time_options);
  const auto count = torch::zeros({}, torch::TensorOptions().device(device).dtype(torch::kInt64));
  const auto overflow = torch::zeros({}, torch::TensorOptions().device(device).dtype(torch::kUInt8));
  const auto reference_out = torch::empty_like(reference);
  constexpr int threads = 256;
  const int blocks = static_cast<int>((pixels + threads - 1) / threads);
  const auto stream = at::cuda::getDefaultCUDAStream(log_before.get_device());
  threshold_events_kernel<<<blocks, threads, 0, stream>>>(
      log_before.data_ptr<float>(), log_after.data_ptr<float>(), reference.data_ptr<float>(),
      reference_out.data_ptr<float>(), events.data_ptr<int32_t>(), event_times.data_ptr<double>(),
      reinterpret_cast<unsigned long long*>(count.data_ptr<int64_t>()), overflow.data_ptr<uint8_t>(),
      pixels, width, start_t, end_t, static_cast<float>(contrast_threshold), max_events_per_pixel);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(events, event_times, count, overflow, reference_out);
}

std::tuple<torch::Tensor, torch::Tensor>
generate_threshold_event_sequence_cuda(
    torch::Tensor log_frames,
    torch::Tensor timestamps,
    double contrast_threshold,
    int64_t max_events_per_pixel,
    double time_origin) {
  TORCH_CHECK(log_frames.is_cuda() && timestamps.is_cuda(), "sequence tensors must be CUDA tensors");
  TORCH_CHECK(log_frames.is_contiguous() && timestamps.is_contiguous(), "sequence tensors must be contiguous");
  TORCH_CHECK(log_frames.dim() == 3 && log_frames.scalar_type() == torch::kFloat,
              "log_frames must have shape [T,H,W] and dtype float32");
  TORCH_CHECK(timestamps.dim() == 1 && timestamps.scalar_type() == torch::kFloat64,
              "timestamps must have shape [T] and dtype float64");
  TORCH_CHECK(log_frames.size(0) == timestamps.size(0) && log_frames.size(0) >= 2,
              "timestamps and log frames must have matching length >= 2");
  const int64_t frames = log_frames.size(0);
  const int64_t height = log_frames.size(1);
  const int64_t width = log_frames.size(2);
  const int64_t pixels = height * width;
  const int64_t intervals = frames - 1;
  TORCH_CHECK(pixels > 0, "empty frames are not supported");
  const auto device = log_frames.device();
  const auto float_options = torch::TensorOptions().device(device).dtype(torch::kFloat32);
  const auto int_options = torch::TensorOptions().device(device).dtype(torch::kInt32);
  const auto references = torch::empty({intervals, pixels}, float_options);
  const auto counts = torch::empty({intervals, pixels}, int_options);
  const auto overflow = torch::zeros({}, torch::TensorOptions().device(device).dtype(torch::kUInt8));
  constexpr int threads = 256;
  const int pixel_blocks = static_cast<int>((pixels + threads - 1) / threads);
  const auto stream = at::cuda::getDefaultCUDAStream(log_frames.get_device());
  count_sequence_kernel<<<pixel_blocks, threads, 0, stream>>>(
      log_frames.data_ptr<float>(), references.data_ptr<float>(), counts.data_ptr<int32_t>(),
      overflow.data_ptr<uint8_t>(), pixels, intervals, static_cast<float>(contrast_threshold), max_events_per_pixel);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const auto flat_counts = counts.reshape({-1});
  const auto offsets = torch::cumsum(flat_counts, 0, torch::kInt64) - flat_counts.to(torch::kInt64);
  const auto total_tensor = torch::sum(flat_counts, torch::kInt64);
  const int64_t total_events = total_tensor.item<int64_t>();
  if (total_events == 0) {
    return std::make_tuple(
        torch::empty({0, 4}, torch::TensorOptions().device(device).dtype(torch::kInt32)), overflow);
  }
  const auto events = torch::empty({total_events, 4}, torch::TensorOptions().device(device).dtype(torch::kInt32));
  const int64_t items = intervals * pixels;
  const int item_blocks = static_cast<int>((items + threads - 1) / threads);
  write_sequence_kernel<<<item_blocks, threads, 0, stream>>>(
      log_frames.data_ptr<float>(), references.data_ptr<float>(), counts.data_ptr<int32_t>(), offsets.data_ptr<int64_t>(),
      timestamps.data_ptr<double>(), events.data_ptr<int32_t>(), pixels, width, intervals,
      static_cast<float>(contrast_threshold), time_origin);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(events, overflow);
}
