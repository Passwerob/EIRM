#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>

namespace {

template <typename scalar_t>
__device__ inline scalar_t scalar_abs(scalar_t x) {
  return x < scalar_t(0) ? -x : x;
}

__device__ inline uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

struct Rng {
  uint64_t state;

  __device__ explicit Rng(uint64_t seed) : state(seed) {}

  __device__ inline uint64_t next_u64() {
    state = splitmix64(state);
    return state;
  }

  template <typename scalar_t>
  __device__ inline scalar_t uniform_open() {
    const uint64_t value = next_u64();
    if constexpr (std::is_same<scalar_t, double>::value) {
      return static_cast<scalar_t>((value >> 11) * 0x1.0p-53) + static_cast<scalar_t>(0x1.0p-54);
    } else {
      return static_cast<scalar_t>((value >> 40) * 0x1.0p-24) + static_cast<scalar_t>(0x1.0p-25);
    }
  }

  template <typename scalar_t>
  __device__ inline scalar_t normal() {
    const scalar_t u1 = uniform_open<scalar_t>();
    const scalar_t u2 = uniform_open<scalar_t>();
    const scalar_t two_pi = static_cast<scalar_t>(6.283185307179586476925286766559);
    return sqrt(static_cast<scalar_t>(-2) * log(u1)) * cos(two_pi * u2);
  }
};

// Acklam's rational approximation. The CUDA kernel only uses it for the
// truncated-normal branch of the official inverse-Gaussian sampler.
template <typename scalar_t>
__device__ inline scalar_t inverse_normal_cdf(scalar_t p) {
  const scalar_t eps = std::numeric_limits<scalar_t>::epsilon();
  p = fmin(fmax(p, eps), static_cast<scalar_t>(1) - eps);
  const scalar_t a1 = static_cast<scalar_t>(-39.6968302866538);
  const scalar_t a2 = static_cast<scalar_t>(220.946098424521);
  const scalar_t a3 = static_cast<scalar_t>(-275.928510446969);
  const scalar_t a4 = static_cast<scalar_t>(138.357751867269);
  const scalar_t a5 = static_cast<scalar_t>(-30.6647980661472);
  const scalar_t a6 = static_cast<scalar_t>(2.50662827745924);
  const scalar_t b1 = static_cast<scalar_t>(-54.4760987982241);
  const scalar_t b2 = static_cast<scalar_t>(161.585836858041);
  const scalar_t b3 = static_cast<scalar_t>(-155.698979859887);
  const scalar_t b4 = static_cast<scalar_t>(66.8013118877197);
  const scalar_t b5 = static_cast<scalar_t>(-13.2806815528857);
  const scalar_t c1 = static_cast<scalar_t>(-0.00778489400243029);
  const scalar_t c2 = static_cast<scalar_t>(-0.322396458041136);
  const scalar_t c3 = static_cast<scalar_t>(-2.40075827716184);
  const scalar_t c4 = static_cast<scalar_t>(-2.54973253934373);
  const scalar_t c5 = static_cast<scalar_t>(4.37466414146497);
  const scalar_t c6 = static_cast<scalar_t>(2.93816398269878);
  const scalar_t d1 = static_cast<scalar_t>(0.00778469570904146);
  const scalar_t d2 = static_cast<scalar_t>(0.32246712907004);
  const scalar_t d3 = static_cast<scalar_t>(2.445134137143);
  const scalar_t d4 = static_cast<scalar_t>(3.75440866190742);
  const scalar_t plow = static_cast<scalar_t>(0.02425);
  const scalar_t phigh = static_cast<scalar_t>(1) - plow;
  if (p < plow) {
    const scalar_t q = sqrt(static_cast<scalar_t>(-2) * log(p));
    return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
           ((((d1 * q + d2) * q + d3) * q + d4) * q + static_cast<scalar_t>(1));
  }
  if (p > phigh) {
    const scalar_t q = sqrt(static_cast<scalar_t>(-2) * log(static_cast<scalar_t>(1) - p));
    return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
            ((((d1 * q + d2) * q + d3) * q + d4) * q + static_cast<scalar_t>(1));
  }
  const scalar_t q = p - static_cast<scalar_t>(0.5);
  const scalar_t r = q * q;
  return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
         (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + static_cast<scalar_t>(1));
}

template <typename scalar_t>
__device__ inline scalar_t sample_inverse_gaussian(scalar_t epsilon, scalar_t c, scalar_t sigma, Rng& rng) {
  const scalar_t tiny = std::numeric_limits<scalar_t>::epsilon();
  scalar_t x;
  if (scalar_abs(c) <= tiny) {
    const scalar_t scale = (epsilon / sigma) * (epsilon / sigma);
    const scalar_t u = rng.uniform_open<scalar_t>();
    const scalar_t normal_tail = inverse_normal_cdf(static_cast<scalar_t>(1) - static_cast<scalar_t>(0.5) * u);
    const scalar_t erfinv = normal_tail * static_cast<scalar_t>(0.7071067811865475244);
    return fmax(scale / (erfinv * erfinv), tiny);
  }

  if (c > static_cast<scalar_t>(0)) {
    x = rng.normal<scalar_t>();
  } else {
    const scalar_t x_max = -sqrt(fmax(static_cast<scalar_t>(0),
                                      static_cast<scalar_t>(-4) * epsilon * c / (sigma * sigma)));
    const scalar_t cdf_max = static_cast<scalar_t>(0.5) * erfc(-x_max * static_cast<scalar_t>(0.7071067811865475244));
    x = inverse_normal_cdf(rng.uniform_open<scalar_t>() * fmax(cdf_max, tiny));
    x = fmin(x, x_max);
  }

  const scalar_t mean = epsilon / c;
  const scalar_t lambda = (epsilon / sigma) * (epsilon / sigma);
  const scalar_t drift = mean / (static_cast<scalar_t>(2) * lambda);
  const scalar_t y = mean * x * x;
  const scalar_t z = fmax(static_cast<scalar_t>(0), static_cast<scalar_t>(4) * lambda * y + y * y);
  const scalar_t candidate = mean + drift * (y - sqrt(z));
  const scalar_t safe_candidate = scalar_abs(candidate) > tiny ? candidate : (candidate < 0 ? -tiny : tiny);
  const scalar_t u = rng.uniform_open<scalar_t>();
  const scalar_t alternate = mean * mean / safe_candidate;
  return fmax(u > mean / (mean + safe_candidate) ? alternate : safe_candidate, tiny);
}

template <typename scalar_t>
__global__ void generate_events_kernel(
    const scalar_t* __restrict__ mu,
    const scalar_t* __restrict__ sigma,
    const scalar_t* __restrict__ residual,
    scalar_t* __restrict__ residual_out,
    int32_t* __restrict__ events,
    unsigned long long* __restrict__ event_count,
    uint8_t* __restrict__ overflow,
    int64_t pixels,
    int64_t width,
    double start_t_double,
    double end_t_double,
    uint64_t seed,
    uint64_t frame_index,
    int64_t max_events_per_pixel) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pixels) return;

  const scalar_t start_t = static_cast<scalar_t>(start_t_double);
  const scalar_t end_t = static_cast<scalar_t>(end_t_double);
  const scalar_t c = mu[index];
  const scalar_t noise = fmax(sigma[index], std::numeric_limits<scalar_t>::epsilon());
  scalar_t residual_value = residual[index];
  scalar_t current_t = start_t;
  const uint64_t stream_seed = seed ^ (static_cast<uint64_t>(index) * 0xd2b74407b1ce6e93ULL) ^
                               (frame_index * 0x9e3779b97f4a7c15ULL);
  Rng rng(stream_seed);
  bool finished = false;

  for (int64_t event_index = 0; event_index < max_events_per_pixel; ++event_index) {
    const scalar_t ep_on = static_cast<scalar_t>(1) - residual_value;
    const scalar_t ep_off = static_cast<scalar_t>(1) + residual_value;
    const scalar_t noise_squared = noise * noise;
    const scalar_t exp_b = exp(static_cast<scalar_t>(2) * c * ep_off / noise_squared);
    const scalar_t exp_a = exp(static_cast<scalar_t>(-2) * c * ep_on / noise_squared);
    scalar_t p_first_on = (exp_b - static_cast<scalar_t>(1)) / (exp_b - exp_a);
    // Match the official torch path: NaN probabilities are replaced by one,
    // then the exact zero-drift case is assigned one half.
    if (isnan(p_first_on)) p_first_on = static_cast<scalar_t>(1);
    if (c == static_cast<scalar_t>(0)) p_first_on = static_cast<scalar_t>(0.5);
    p_first_on = fmin(fmax(p_first_on, static_cast<scalar_t>(0)), static_cast<scalar_t>(1));
    const bool on = rng.uniform_open<scalar_t>() <= p_first_on;
    const scalar_t epsilon = on ? ep_on : ep_off;
    const scalar_t drift = on ? c : -c;
    const scalar_t delta_t = sample_inverse_gaussian(epsilon, drift, noise, rng);
    const scalar_t event_t = current_t + delta_t;

    if (!(event_t < end_t) || !isfinite(event_t)) {
      const scalar_t signed_epsilon = (on ? static_cast<scalar_t>(1) : static_cast<scalar_t>(-1)) * epsilon;
      residual_value += signed_epsilon * (end_t - current_t) / delta_t;
      finished = true;
      break;
    }

    const int64_t slot = static_cast<int64_t>(atomicAdd(event_count, 1ULL));
    if (slot < pixels * max_events_per_pixel) {
      // The official simulator rounds timestamps to integer microseconds
      // before serializing events.
      events[slot * 4 + 0] = static_cast<int32_t>(round(static_cast<double>(event_t)));
      events[slot * 4 + 1] = static_cast<int32_t>(index % width);
      events[slot * 4 + 2] = static_cast<int32_t>(index / width);
      events[slot * 4 + 3] = on ? 1 : -1;
    } else {
      *overflow = 1;
    }
    residual_value = static_cast<scalar_t>(0);
    current_t = event_t;
  }

  residual_out[index] = residual_value;
  if (!finished) *overflow = 1;
}

}  // namespace

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
    int64_t max_events_per_pixel) {
  const auto pixels = height * width;
  TORCH_CHECK(pixels > 0, "empty frame is not supported");
  TORCH_CHECK(max_events_per_pixel <= std::numeric_limits<int64_t>::max() / pixels,
              "max_events_per_pixel is too large");
  const auto capacity = pixels * max_events_per_pixel;
  TORCH_CHECK(end_t <= static_cast<double>(std::numeric_limits<int32_t>::max()),
              "dvs_fast event timestamps must fit signed int32 microseconds");
  auto output_options = torch::TensorOptions().device(mu.device()).dtype(torch::kInt32);
  auto events = torch::empty({capacity, 4}, output_options);
  auto count = torch::zeros({}, torch::TensorOptions().device(mu.device()).dtype(torch::kInt64));
  auto overflow = torch::zeros({}, torch::TensorOptions().device(mu.device()).dtype(torch::kUInt8));
  auto residual_out = torch::empty_like(residual);

  const int threads = 256;
  const int blocks = static_cast<int>((pixels + threads - 1) / threads);
  auto stream = at::cuda::getDefaultCUDAStream(mu.device().index());
  AT_DISPATCH_FLOATING_TYPES(mu.scalar_type(), "dvs_fast_generate_events", [&] {
    generate_events_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        mu.data_ptr<scalar_t>(), sigma.data_ptr<scalar_t>(), residual.data_ptr<scalar_t>(),
        residual_out.data_ptr<scalar_t>(), events.data_ptr<int32_t>(),
        reinterpret_cast<unsigned long long*>(count.data_ptr<int64_t>()),
        overflow.data_ptr<uint8_t>(), pixels, width, start_t, end_t,
        static_cast<uint64_t>(seed), static_cast<uint64_t>(frame_index), max_events_per_pixel);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(events, count, overflow, residual_out);
}
