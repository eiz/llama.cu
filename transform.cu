#define LLAMA_CU_IMPLEMENTATION
#include "llama_cu.h"

#include <cuda_fp16.h>

namespace llama_cu {
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

template <typename T>
bool load_tensor_by_address(tensor4d<T>& output,
                            std::string const& input_model,
                            llama_params const& params,
                            std::string const& tensor_name,
                            std::string const& part_name,
                            int layer_index) {
  size_t h, w;
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
  if (part_name == "scale") {
    if (params.q_type == quantization_type::fp16) {
      fprintf(stderr, "Cannot load scale for fp16 model\n");
      return false;
    }
    w /= params.quantization_block_size;
  }
  mapped_buffer buf("models/" + input_model + "/" +
                    (layer_index >= 0 ? "layers." + std::to_string(layer_index) + "." : "") +
                    tensor_name + "." + part_name + "__" +
                    ((h > 1) ? std::to_string(h) + "_" : "") + std::to_string(w));
  if (!buf.is_ok()) {
    fprintf(stderr, "Failed to load tensor %s\n", tensor_name.c_str());
    return false;
  }
  output = tensor4d<T>(1, 1, h, w);
  output.gpu.copy_from(buf);
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
  tensor4d<uint8_t> weights;
  if (!load_tensor_by_address(weights, input_model, params, tensor_name, "weight", layer_index)) {
    fprintf(stderr, "Failed to load weights for %s\n", tensor_name.c_str());
  }
  uint8_t* yes_this_is_silly = reinterpret_cast<uint8_t*>(malloc(weights.gpu.size()));
  weights.gpu.copy_to(yes_this_is_silly, weights.gpu.size());
  uint32_t histogram[256] = {};
  for (size_t i = 0; i < weights.gpu.size(); ++i) {
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

bool do_svd(const std::string& input_model,
            const llama_params& params,
            const std::string& tensor_name,
            int layer_index) {
  tensor4d<uint8_t> weights;

  if (!load_tensor_by_address(weights, input_model, params, tensor_name, "weight", layer_index)) {
    fprintf(stderr, "Failed to load weights for %s\n", tensor_name.c_str());
  }

  // TODO
  return false;
}

}  // namespace llama_cu

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
  llama_cu::mapped_buffer in_params_buf("models/" + input_model_name + "/params");
  if (!in_params_buf.is_ok()) {
    fprintf(stderr, "Could not load params file\n");
    return 1;
  }
  llama_cu::llama_params params;
  if (!params.load(in_params_buf)) {
    fprintf(stderr, "Invalid model parameters\n");
    return 1;
  }
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
  } else if (operation == "svd") {
    std::string tensor_name = "attention.wq";
    if (argc > 3) {
      tensor_name = argv[3];
    }
    int layer_index = 0;
    if (argc > 4) {
      layer_index = atoi(argv[4]);
    }
    return do_svd(input_model_name, params, tensor_name, layer_index) == false;
  } else {
    fprintf(stderr, "Unknown operation: %s\n", operation.c_str());
  }
  return 0;
}
