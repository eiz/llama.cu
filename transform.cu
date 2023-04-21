#define LLAMA_CU_IMPLEMENTATION
#include "llama_cu.h"
#undef NDEBUG

#include <cassert>

#ifdef ENABLE_CUBLAS
#include <cublas_v2.h>
#endif

#ifdef ENABLE_CUSOLVER
#include <cusolverDn.h>
#endif

#include <cuda_fp16.h>

namespace llama_cu {
// TODO: bad kernel (no tiling)
__global__ void half2float_transpose(float* output, half* input, int h, int w) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) {
    output[x * h + y] = __half2float(input[y * w + x]);
  }
}

// atm there's no reason to do this on the gpu, but i want it for later
__global__ void quantize_absmax_uint8(uint8_t* output,
                                      half* scales,
                                      half const* __restrict__ input,
                                      int M,
                                      int N,
                                      int K) {
  __shared__ float s_warp_reduced[32];
  __shared__ half s_absmax;
  float absmax_val = 0;
  int row = blockIdx.y * blockDim.y;
  int tid = threadIdx.x;
  int lane_id = tid % 32;
  int warp_id = tid / 32;
  bool warp_leader = lane_id == 0;
  int block_count = N / K;
  for (int i = blockIdx.x * K + tid; i < (blockIdx.x + 1) * K; i += blockDim.x) {
    float val = __float2half(input[row * N + i]);
    absmax_val = fmax(absmax_val, fabs(val));
  }
  __syncthreads();
  for (int i = 16; i > 0; i /= 2) {
    absmax_val = fmax(absmax_val, __shfl_xor_sync(~0, absmax_val, i));
  }
  if (warp_leader) {
    s_warp_reduced[warp_id] = absmax_val;
  }
  __syncthreads();
  if (warp_id == 0) {
    absmax_val = lane_id < (blockDim.x / 32) ? s_warp_reduced[lane_id] : 0;
    for (int i = 16; i > 0; i /= 2) {
      absmax_val = fmax(absmax_val, __shfl_xor_sync(~0, absmax_val, i));
    }
    if (warp_leader) {
      s_absmax = absmax_val / 127.5f;
    }
  }
  __syncthreads();
  if (tid == 0) {
    scales[row * block_count + blockIdx.x] = s_absmax;
  }
  for (int i = blockIdx.x * K + tid; i < (blockIdx.x + 1) * K; i += blockDim.x) {
    output[row * N + i] = static_cast<uint8_t>(
        round(127.5f + __half2float(input[row * N + i]) / __half2float(s_absmax)));
  }
}

__global__ void dequantize_absmax_uint8(half* output,
                                        half const* __restrict__ scales,
                                        uint8_t const* __restrict__ input,
                                        int M,
                                        int N,
                                        int K) {
  int row = blockIdx.y;
  int tid = threadIdx.x;
  int block_count = N / K;
  for (int i = blockIdx.x * K + tid; i < (blockIdx.x + 1) * K; i += blockDim.x) {
    output[row * N + i] = __float2half((float(input[row * N + i]) - 127.5f) *
                                       __half2float(scales[row * block_count + blockIdx.x]));
  }
}

// https://stackoverflow.com/a/51549250
// technically this does not handle nan properly but those dont exist right
// actually my sums are always positive so you dont even need the negative case
__device__ __forceinline__ float atomic_max_float(float* addr, float value) {
  float old;
  old = (value >= 0) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
                     : __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
  return old;
}

// output[0] = mean squared error, output[1] = max error, output[2] = mean absolute error
__global__ void error_stats(float* output,
                            half const* __restrict__ lhs,
                            half const* __restrict__ rhs,
                            int length) {
  __shared__ float s_sum_sq_reduced[32];
  __shared__ float s_max_reduced[32];
  __shared__ float s_sum_reduced[32];
  float sum_sq = 0;
  float max = 0;
  float sum = 0;
  int tid = threadIdx.x;
  int lane_id = tid % 32;
  int warp_id = tid / 32;
  bool warp_leader = lane_id == 0;
  int block_size = length / gridDim.x;
  int block_start = block_size * blockIdx.x;
  int block_end = block_size * (blockIdx.x + 1);
  for (int i = block_start + tid; i < block_end && i < length; i += blockDim.x) {
    float diff = __half2float(lhs[i]) - __half2float(rhs[i]);
    sum_sq += diff * diff;
    max = fmax(max, fabs(diff));
    sum += fabs(diff);
  }
  __syncthreads();
  for (int i = 16; i > 0; i /= 2) {
    sum_sq += __shfl_xor_sync(~0, sum_sq, i);
    max = fmax(max, __shfl_xor_sync(~0, max, i));
    sum += __shfl_xor_sync(~0, sum, i);
  }
  if (warp_leader) {
    s_sum_sq_reduced[warp_id] = sum_sq;
    s_max_reduced[warp_id] = max;
    s_sum_reduced[warp_id] = sum;
  }
  __syncthreads();
  if (warp_id == 0) {
    sum_sq = lane_id < (blockDim.x / 32) ? s_sum_sq_reduced[lane_id] : 0;
    max = lane_id < (blockDim.x / 32) ? s_max_reduced[lane_id] : 0;
    sum = lane_id < (blockDim.x / 32) ? s_sum_reduced[lane_id] : 0;
    for (int i = 16; i > 0; i /= 2) {
      sum_sq += __shfl_xor_sync(~0, sum_sq, i);
      max = fmax(max, __shfl_xor_sync(~0, max, i));
      sum += __shfl_xor_sync(~0, sum, i);
    }
    if (warp_leader) {
      atomicAdd(output, sum_sq / length);
      atomic_max_float(output + 1, max);
      atomicAdd(output + 2, sum / length);
    }
  }
}

bool quantize_one(std::string const& input_model,
                  std::string const& output_model,
                  std::string const& name,
                  size_t h,
                  size_t w,
                  int block_size) {
  printf("quantizing %s (%zu x %zu)\n", name.c_str(), h, w);
  tensor4d<half> input(1, 1, h, w);
  tensor4d<half> scales(1, 1, h, w / block_size);
  tensor4d<uint8_t> output(1, 1, h, w);
  tensor4d<half> output_dequantized(1, 1, h, w);
  tensor4d<float> error(1, 1, 1, 3);
  mapped_buffer buf("models/" + input_model + "/" + name + ".weight__" + std::to_string(h) + "_" +
                    std::to_string(w));
  if (!buf.is_ok() || buf.size() != input.gpu.size()) {
    fprintf(stderr, "failed to load tensor %s (%zu x %zu)\n", name.c_str(), h, w);
    return false;
  }
  input.gpu.copy_from(buf);
  quantize_absmax_uint8<<<dim3(w / block_size, h), std::min(block_size, 1024)>>>(
      output.data(), scales.data(), input.data(), h, w, block_size);
  dequantize_absmax_uint8<<<dim3(w / block_size, h), std::min(block_size, 1024)>>>(
      output_dequantized.data(), scales.data(), output.data(), h, w, block_size);
  error_stats<<<128, 1024>>>(error.data(), input.data(), output_dequantized.data(), h * w);
  float host_error[3];
  error.gpu.copy_to(&host_error, sizeof(float) * 3);
  printf("  rms/max/mean error: %f/%f/%f\n", sqrt(host_error[0]), host_error[1], host_error[2]);
  mapped_buffer out_buf("models/" + output_model + "/" + name + ".weight__" + std::to_string(h) +
                            "_" + std::to_string(w),
                        output.gpu.size());
  mapped_buffer scale_buf("models/" + output_model + "/" + name + ".scale__" + std::to_string(h) +
                              "_" + std::to_string(w / block_size),
                          scales.gpu.size());
  if (!out_buf.is_ok()) {
    fprintf(stderr, "failed to create weight tensor %s (%zu x %zu)\n", name.c_str(), h, w);
    return false;
  }
  if (!scale_buf.is_ok()) {
    fprintf(stderr, "failed to create scale tensor %s (%zu x %zu)\n", name.c_str(), h, w);
    return false;
  }
  scales.gpu.copy_to(scale_buf.data(), scale_buf.size());
  output.gpu.copy_to(out_buf.data(), out_buf.size());
  return true;
}

bool copy_tensor(std::string const& input_model,
                 std::string const& output_model,
                 std::string const& name,
                 size_t w) {
  mapped_buffer in_buf("models/" + input_model + "/" + name + ".weight__" + std::to_string(w));
  mapped_buffer out_buf("models/" + output_model + "/" + name + ".weight__" + std::to_string(w),
                        in_buf.size());

  if (!in_buf.is_ok()) {
    fprintf(stderr, "failed to load tensor %s (%zu)\n", name.c_str(), w);
    return false;
  }

  if (!out_buf.is_ok()) {
    fprintf(stderr, "failed to create tensor %s (%zu)\n", name.c_str(), w);
    return false;
  }

  memcpy(out_buf.data(), in_buf.data(), in_buf.size());
  return true;
}

bool do_quantize(const std::string& input_model,
                 const std::string& output_model,
                 const llama_params& in_params,
                 int block_size) {
  if (in_params.q_type != quantization_type::fp16) {
    fprintf(stderr, "Input model is not fp16\n");
    return false;
  }
  RETURN_UNLESS(quantize_one(input_model, output_model, "tok_embeddings", in_params.n_vocab,
                             in_params.dim, block_size));
  RETURN_UNLESS(quantize_one(input_model, output_model, "output", in_params.n_vocab, in_params.dim,
                             block_size));
  RETURN_UNLESS(copy_tensor(input_model, output_model, "norm", in_params.dim));
  for (int i = 0; i < in_params.n_layers; ++i) {
    std::string basename = "layers." + std::to_string(i) + ".";
    RETURN_UNLESS(
        copy_tensor(input_model, output_model, basename + "attention_norm", in_params.dim));
    RETURN_UNLESS(copy_tensor(input_model, output_model, basename + "ffn_norm", in_params.dim));
    RETURN_UNLESS(quantize_one(input_model, output_model, basename + "attention.wq", in_params.dim,
                               in_params.dim, block_size));
    RETURN_UNLESS(quantize_one(input_model, output_model, basename + "attention.wk", in_params.dim,
                               in_params.dim, block_size));
    RETURN_UNLESS(quantize_one(input_model, output_model, basename + "attention.wv", in_params.dim,
                               in_params.dim, block_size));
    RETURN_UNLESS(quantize_one(input_model, output_model, basename + "attention.wo", in_params.dim,
                               in_params.dim, block_size));
    RETURN_UNLESS(quantize_one(input_model, output_model, basename + "feed_forward.w1",
                               in_params.ffn_dim(), in_params.dim, block_size));
    RETURN_UNLESS(quantize_one(input_model, output_model, basename + "feed_forward.w2",
                               in_params.dim, in_params.ffn_dim(), block_size));
    RETURN_UNLESS(quantize_one(input_model, output_model, basename + "feed_forward.w3",
                               in_params.ffn_dim(), in_params.dim, block_size));
  }
  llama_params out_params = in_params;
  out_params.q_type = quantization_type::uint8;
  out_params.quantization_block_size = block_size;
  mapped_buffer params_buf("models/" + output_model + "/params", sizeof(llama_params));
  if (!params_buf.is_ok()) {
    fprintf(stderr, "Failed to create params file\n");
    return false;
  }
  memcpy(params_buf.data(), &out_params, sizeof(llama_params));
  return true;
}

bool load_tensor_by_address(generic_tensor& output,
                            std::string const& input_model,
                            llama_params const& params,
                            std::string const& tensor_name,
                            int layer_index) {
  size_t h, w;
  bool is_quantized = params.q_type != quantization_type::fp16;
  if (tensor_name == "tok_embeddings" || tensor_name == "output") {
    h = params.n_vocab;
    w = params.dim;
    if (layer_index != -1) {
      fprintf(stderr, "Invalid layer index %d for tensor %s\n", layer_index, tensor_name.c_str());
      return false;
    }
  } else if (tensor_name == "attention.wq" || tensor_name == "attention.wk" ||
             tensor_name == "attention.wv" || tensor_name == "attention.wo") {
    h = w = params.dim;
  } else if (tensor_name == "feed_forward.w1" || tensor_name == "feed_forward.w3") {
    h = params.ffn_dim();
    w = params.dim;
  } else if (tensor_name == "feed_forward.w2") {
    h = params.dim;
    w = params.ffn_dim();
  } else {
    // TODO: norm weights
    fprintf(stderr, "Unknown tensor name %s\n", tensor_name.c_str());
    return false;
  }
  auto prefix = "models/" + input_model + "/" +
                (layer_index >= 0 ? "layers." + std::to_string(layer_index) + "." : "") +
                tensor_name + ".";
  auto suffix = "__" + ((h > 1) ? std::to_string(h) + "_" : "") + std::to_string(w);
  mapped_buffer weight_buf(prefix + "weight" + suffix);
  mapped_buffer scale_buf;
  if (!weight_buf.is_ok()) {
    fprintf(stderr, "Failed to load tensor %s\n", tensor_name.c_str());
    return false;
  }
  if (is_quantized) {
    scale_buf = mapped_buffer(prefix + "scale" + suffix);
    if (!scale_buf.is_ok()) {
      fprintf(stderr, "Failed to load tensor %s\n", tensor_name.c_str());
      return false;
    }
    quantized_tensor quant;
    tensor4d<uint8_t> t_values(1, 1, h, w);
    t_values.gpu.copy_from(weight_buf);
    tensor4d<half> t_scales(1, 1, h, w / params.quantization_block_size);
    t_scales.gpu.copy_from(scale_buf);
    quant.q_type = params.q_type;
    quant.q_values = std::move(t_values);
    quant.scales = std::move(t_scales);
    output = std::move(quant);
  } else {
    tensor4d<half> t_result(1, 1, h, w);
    t_result.gpu.copy_from(weight_buf);
    output = std::move(t_result);
  }
  return true;
}

bool do_histogram(const std::string& input_model,
                  const llama_params& params,
                  const std::string& tensor_name,
                  int layer_index) {
  if (params.q_type != quantization_type::uint8) {
    fprintf(stderr, "Input model is not uint8\n");
    return false;
  }
  generic_tensor weights;
  if (!load_tensor_by_address(weights, input_model, params, tensor_name, layer_index)) {
    fprintf(stderr, "Failed to load weights for %s\n", tensor_name.c_str());
  }
  if (!weights.is_quantized()) {
    fprintf(stderr, "Input tensor is not quantized\n");
    return false;
  }
  auto& quant = weights.as_quantized();
  uint8_t* yes_this_is_silly = reinterpret_cast<uint8_t*>(malloc(quant.q_values.gpu.size()));
  quant.q_values.gpu.copy_to(yes_this_is_silly, quant.q_values.gpu.size());
  uint32_t histogram[256] = {};
  for (size_t i = 0; i < quant.q_values.gpu.size(); ++i) {
    histogram[yes_this_is_silly[i]]++;
  }
  printf("      ");
  for (int i = 0; i < 16; ++i) {
    printf("%-8X", i);
  }
  printf("\n");
  for (int i = 0; i < 16; ++i) {
    printf("%4X: ", i);
    for (int j = 0; j < 16; ++j) {
      printf("%-8d", histogram[i * 16 + j]);
    }
    printf("\n");
  }
  free(yes_this_is_silly);
  return true;
}

#if defined(ENABLE_CUBLAS) && defined(ENABLE_CUSOLVER)
void dequantize(generic_tensor& input) {
  if (input.is_quantized()) {
    auto& quant = input.as_quantized();
    tensor4d<half> output_dequantized(1, 1, quant.q_values.h, quant.q_values.w);
    dequantize_absmax_uint8<<<dim3(quant.q_values.w / quant.quantization_block_size,
                                   quant.q_values.h),
                              std::min<int>(quant.quantization_block_size, 1024)>>>(
        output_dequantized.data(), quant.scales.data(), quant.q_values.data(), quant.q_values.h,
        quant.q_values.w, quant.quantization_block_size);
    input = std::move(output_dequantized);
  }
}

void svd(tensor4d<float>& out_u,
         tensor4d<float>& out_v,
         tensor4d<float>& out_s,
         generic_tensor& input) {
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  cusolverDnHandle_t cusolver_handle;
  cusolverDnCreate(&cusolver_handle);
  dequantize(input);
  assert(input.is_fp16());
  auto& t_weights = input.as_fp16();
  tensor4d<float> t_weights_transposed(1, 1, t_weights.w, t_weights.h);
  half2float_transpose<<<dim3(t_weights.w / 32, t_weights.h / 32), dim3(32, 32)>>>(
      t_weights_transposed.data(), t_weights.data(), t_weights.h, t_weights.w);
  tensor4d<float> t_u(1, 1, t_weights.h, t_weights.h);
  tensor4d<float> t_v(1, 1, t_weights.w, t_weights.w);
  tensor4d<float> t_s(1, 1, std::min(t_weights.h, t_weights.w), 1);
  tensor4d<float> rwork(1, 1, 1, std::min(t_weights.h, t_weights.w) - 1);
  int lwork = 0;
  tensor4d<int> dev_info(1, 1, 1, 1);
  cusolverDnSgesvd_bufferSize(cusolver_handle, t_weights.w, t_weights.h, &lwork);
  tensor4d<float> d_work(1, 1, 1, lwork);
  cusolverStatus_t status;
  status = cusolverDnSgesvd(cusolver_handle, 'A', 'A', t_weights.h, t_weights.w,
                            t_weights_transposed.data(), t_weights.h, t_s.data(), t_u.data(), t_u.h,
                            t_v.data(), t_v.h, d_work.data(), lwork, rwork.data(), dev_info.data());
  if (status != 0) {
    fprintf(stderr, "SVD failed with status %d\n", status);
    exit(1);
  }
  cusolverDnDestroy(cusolver_handle);
  cublasDestroy(cublas_handle);
  out_u = std::move(t_u);
  out_v = std::move(t_v);
  out_s = std::move(t_s);
}

void print_svd(tensor4d<float>& t_s) {
  float* host_data = (float*)malloc(sizeof(float) * t_s.h);
  t_s.gpu.copy_to(host_data, sizeof(float) * t_s.h);
  float sum = 0.0;
  float cumsum = 0.0;
  for (int i = 0; i < t_s.h; ++i) {
    sum += host_data[i];
  }
  for (int i = 0; i < t_s.h; ++i) {
    cumsum += host_data[i];
    printf("%5d: %f (so far: %.2f%%)\n", i, host_data[i], cumsum / sum * 100.0);
  }
  free(host_data);
}

bool do_svd(const std::string& input_model,
            const llama_params& params,
            const std::string& tensor_name,
            int layer_index) {
  generic_tensor weights;
  if (!load_tensor_by_address(weights, input_model, params, tensor_name, layer_index)) {
    fprintf(stderr, "Failed to load weights for %s\n", tensor_name.c_str());
    return false;
  }
  tensor4d<float> t_u, t_v, t_s;
  svd(t_u, t_v, t_s, weights);
  print_svd(t_s);
  return true;
}

#define ELEMENTWISE_BINOP(name, expr)                                             \
  namespace kernel {                                                              \
  template <typename T>                                                           \
  __global__ void elementwise_##name(T* a, T* b, size_t size) {                   \
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;                             \
    if (i < size) {                                                               \
      a[i] = expr;                                                                \
    }                                                                             \
  }                                                                               \
  }                                                                               \
  template <typename T>                                                           \
  void elementwise_##name(tensor4d<T>& a, tensor4d<T>& b) {                       \
    assert(a.h == b.h);                                                           \
    assert(a.w == b.w);                                                           \
    auto n = a.h * a.w;                                                           \
    kernel::elementwise_##name<<<ceil_div(n, 256), 256>>>(a.data(), b.data(), n); \
  }
ELEMENTWISE_BINOP(add, a[i] + b[i])
ELEMENTWISE_BINOP(sub, a[i] - b[i])
ELEMENTWISE_BINOP(mul, a[i] * b[i])

bool do_compare_svd(const std::string& base_model,
                    const std::string& finetune_model,
                    const llama_params& base_params,
                    const llama_params& finetune_params,
                    const std::string& tensor_name,
                    int layer_index) {
  printf("comparing %s/%s, layer %d, tensor %s\n", base_model.c_str(), finetune_model.c_str(),
         layer_index, tensor_name.c_str());
  generic_tensor base_weights, finetune_weights;
  if (!load_tensor_by_address(base_weights, base_model, base_params, tensor_name, layer_index)) {
    fprintf(stderr, "Failed to load base weights for %s\n", tensor_name.c_str());
    return false;
  }
  if (!load_tensor_by_address(finetune_weights, finetune_model, finetune_params, tensor_name,
                              layer_index)) {
    fprintf(stderr, "Failed to load finetune weights for %s\n", tensor_name.c_str());
    return false;
  }
  dequantize(base_weights);
  dequantize(finetune_weights);
  base_weights.as_fp16().view().debug_print("base_weights");
  finetune_weights.as_fp16().view().debug_print("finetune_weights");
  elementwise_sub(finetune_weights.as_fp16(), base_weights.as_fp16());
  finetune_weights.as_fp16().view().debug_print("finetune_weights - base_weights");
  tensor4d<float> t_u, t_v, t_s;
  svd(t_u, t_v, t_s, finetune_weights);
  print_svd(t_s);
  return true;
}
#endif  // ENABLE_CUBLAS && ENABLE_CUSOLVER

}  // namespace llama_cu

llama_cu::llama_params load_params(std::string const& input_model_name) {
  llama_cu::mapped_buffer in_params_buf("models/" + input_model_name + "/params");
  if (!in_params_buf.is_ok()) {
    fprintf(stderr, "Could not load params file\n");
    exit(1);
  }
  llama_cu::llama_params params;
  if (!params.load(in_params_buf)) {
    fprintf(stderr, "Invalid model parameters\n");
    exit(1);
  }
  return params;
}

int main(int argc, char** argv) {
  llama_cu::initialize();
  std::string operation = "quantize";
  if (argc > 1) {
    operation = argv[1];
  }
  std::string input_model_name = "filthy_instruct_v6";
  if (argc > 2) {
    input_model_name = argv[2];
  }
  llama_cu::llama_params params = load_params(input_model_name);
  if (operation == "quantize") {
    std::string output_model_name = "filthy_instruct_v6_q8";
    if (argc > 3) {
      output_model_name = argv[3];
    }
    int block_size = 128;
    if (argc > 4) {
      block_size = atoi(argv[4]);
    }
    return do_quantize(input_model_name, output_model_name, params, block_size) == false;
  } else if (operation == "histogram") {
    std::string tensor_name = "attention.wq";
    if (argc > 3) {
      tensor_name = argv[3];
    }
    int layer_index = 0;
    if (argc > 4) {
      layer_index = atoi(argv[4]);
    }
    return do_histogram(input_model_name, params, tensor_name, layer_index) == false;
  }
#if defined(ENABLE_CUBLAS) && defined(ENABLE_CUSOLVER)
  else if (operation == "svd") {
    std::string tensor_name = "attention.wq";
    if (argc > 3) {
      tensor_name = argv[3];
    }
    int layer_index = 0;
    if (argc > 4) {
      layer_index = atoi(argv[4]);
    }
    return do_svd(input_model_name, params, tensor_name, layer_index) == false;
  } else if (operation == "compare_svd") {
    std::string finetune_model_name = "filthy_instruct_v6";
    if (argc > 3) {
      finetune_model_name = argv[3];
    }
    llama_cu::llama_params finetune_params = load_params(finetune_model_name);
    std::string tensor_name = "attention.wq";
    if (argc > 4) {
      tensor_name = argv[4];
    }
    int layer_index = 0;
    if (argc > 5) {
      layer_index = atoi(argv[5]);
    }
    return do_compare_svd(input_model_name, finetune_model_name, params, finetune_params,
                          tensor_name, layer_index) == false;
  }
#endif  // ENABLE_CUBLAS && ENABLE_CUSOLVER
  else {
    fprintf(stderr, "Unknown operation: %s\n", operation.c_str());
  }
  return 0;
}
