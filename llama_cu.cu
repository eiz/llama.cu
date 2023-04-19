#define LLAMA_CU_IMPLEMENTATION
#include "llama_cu.h"
#undef NDEBUG
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef ENABLE_CUBLAS
#include <cublas_v2.h>
#endif

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <mma.h>

namespace llama_cu {
int matmul_type = 4;

__device__ __host__ __forceinline__ size_t ceil_div(size_t a, size_t b) {
  return (a + b - 1) / b;
}

__device__ __host__ __forceinline__ size_t round_up(size_t a, size_t b) {
  return ceil_div(a, b) * b;
}

namespace kernel {
// TODO: these embed kernels are objectively terrible
__global__ void embed(int n_dim,
                      int n_ids,
                      short* in_ids,
                      __half* out_embeddings,
                      __half* embeddings_table) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n_ids) {
    int id = in_ids[tid];
    for (int i = 0; i < n_dim; i++) {
      out_embeddings[tid * n_dim + i] = embeddings_table[id * n_dim + i];
    }
  }
}

__global__ void embed_uint8(int n_dim,
                            int n_ids,
                            short* in_ids,
                            __half* out_embeddings,
                            uint8_t* embeddings_table,
                            half* scales,
                            int block_size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int block_count = n_dim / block_size;
  if (tid < n_ids) {
    int id = in_ids[tid];
    for (int i = 0; i < n_dim; i++) {
      out_embeddings[tid * n_dim + i] = (float(embeddings_table[id * n_dim + i]) - 127.5f) *
                                        __half2float(scales[id * block_count + i / block_size]);
    }
  }
}

// row-wise rms normalization
// 1 block per row, x = row, 8 warps per block
__global__ void rms_norm(__half* output, __half* input, __half* weights, int h, int w, float eps) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int row_idx = row * w;
  int warp_id = tid / 32;
  bool warp_leader = (tid % 32) == 0;
  __shared__ float s_rms_inv;
  __shared__ float s_warp_reduced[8];
  float sum_val = 0.0f;
  // sum_sq: thread reduction
  for (int i = tid; i < w; i += blockDim.x) {
    float val = __half2float(input[row_idx + i]);
    sum_val += val * val;
  }
  __syncthreads();
  // sum_sq: warp reduction
  for (int offset = 16; offset > 0; offset /= 2) {
    float other_val = __shfl_xor_sync(~0, sum_val, offset);
    sum_val += other_val;
  }
  if (warp_leader) {
    s_warp_reduced[warp_id] = sum_val;
  }
  // sum_sq: block reduction
  __syncthreads();
  if (warp_id == 0) {
    sum_val = (tid < 8) ? s_warp_reduced[tid] : 0.0f;
    for (int offset = 4; offset > 0; offset /= 2) {
      float other_val = __shfl_xor_sync(~0, sum_val, offset);
      sum_val += other_val;
    }
    if (warp_leader) {
      s_rms_inv = rsqrt((sum_val / w) + eps);
    }
  }
  __syncthreads();
  float rms_inv = s_rms_inv;
  for (int i = tid; i < w; i += blockDim.x) {
    float val = __half2float(input[row_idx + i]);
    output[row_idx + i] = weights[i] * __float2half(val * rms_inv);
  }
}

// output = lhs * rhs^T; lhs = (m, p); rhs = (n, p); output = (m, n)
// this only exists as a trivial reference point for optimized kernels.
__global__ void
matmul_nt(__half* output, __half* lhs, __half* rhs, int m, int p, int n, float beta = 0.0f) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r < m && c < n) {
    float sum = 0;
    for (int i = 0; i < p; i++) {
      sum += __half2float(lhs[r * p + i]) * __half2float(rhs[c * p + i]);
    }
    output[r * n + c] = sum + beta * __half2float(output[r * n + c]);
  }
}

// the simplest quantized right side kernel. reference.
__global__ void matmul_nt_fp16u8(half* __restrict__ output,
                                 half const* __restrict__ lhs,
                                 uint8_t const* __restrict__ rhs,
                                 half const* __restrict__ scales,
                                 int m,
                                 int p,
                                 int n,
                                 int block_size,
                                 float beta = 0.0f) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int block_count = p / block_size;
  if (r < m && c < n) {
    float sum = 0;
    for (int i = 0; i < p; i++) {
      sum += __half2float(lhs[r * p + i]) * (float(rhs[c * p + i]) - 127.5) *
             __half2float(scales[c * block_count + i / block_size]);
    }
    output[r * n + c] = sum + beta * __half2float(output[r * n + c]);
  }
}

// output = lhs * rhs^T; lhs = (m, p); rhs = (n, p); output = (m, n)
// blockDim must be (256, 1, 1). gridDim must be (m/128 * n/64, 1, 1)
// m must be a multiple of 128, n and p must be multiples of 64
__global__ void matmul_nt_wmma_128x64x64(__half* __restrict__ output,
                                         __half const* __restrict__ lhs,
                                         __half const* __restrict__ rhs,
                                         int m,
                                         int p,
                                         int n,
                                         float beta = 0.0f) {
  using namespace nvcuda::wmma;
  extern __shared__ void* sdata[];
  const int SDATA_BASE_LHS = 0;
  const int SDATA_BASE_RHS = 128 * 80;
#define SDATA(type, side, stride, d0, d1) \
  (((type*)sdata)[SDATA_BASE_##side + ((d0) * (stride)) + (d1)])
#define LHS(d0, d1) SDATA(__half, LHS, 80, d0, d1)
#define RHS(d0, d1) SDATA(__half, RHS, 80, d0, d1)
#define OUT(d0, d1) SDATA(float, LHS, 64, d0, d1)
  int bid = blockIdx.x;
  int dim_y = m / 128;
  int bx = (bid / dim_y) * 64, by = (bid % dim_y) * 128;
  unsigned tid = threadIdx.x;
  int tlo = tid & 15, thi = tid >> 4;
  int warp_id = tid / 32;
  int wx = 32 * (warp_id & 1);
  int wy = 32 * (warp_id >> 1);
  fragment<accumulator, 16, 16, 16, float> frag_accum[2][2];
  fragment<matrix_a, 16, 16, 16, __half, row_major> frag_lhs[2];
  fragment<matrix_b, 16, 16, 16, __half, col_major> frag_rhs[2];
  fill_fragment(frag_accum[0][0], 0.0f);
  fill_fragment(frag_accum[0][1], 0.0f);
  fill_fragment(frag_accum[1][0], 0.0f);
  fill_fragment(frag_accum[1][1], 0.0f);
  for (int t = 0; t < p; t += 64) {
    for (int i = 0; i < 8; ++i) {
      int lhs_idx = (by + thi * 8 + i) * p + t + tlo * 4;
      *((short4*)&LHS(thi * 8 + i, tlo * 4)) = *(short4*)(&lhs[lhs_idx]);
    }
    for (int i = 0; i < 4; ++i) {
      int rhs_idx = (bx + thi * 4 + i) * p + t + tlo * 4;
      *((short4*)&RHS(thi * 4 + i, tlo * 4)) = *(short4*)(&rhs[rhs_idx]);
    }
    __syncthreads();
    for (int i = 0; i < 64; i += 16) {
      load_matrix_sync(frag_lhs[0], &LHS(wy, i), 80);
      load_matrix_sync(frag_rhs[0], &RHS(wx, i), 80);
      mma_sync(frag_accum[0][0], frag_lhs[0], frag_rhs[0], frag_accum[0][0]);
      load_matrix_sync(frag_rhs[1], &RHS(wx + 16, i), 80);
      mma_sync(frag_accum[0][1], frag_lhs[0], frag_rhs[1], frag_accum[0][1]);
      load_matrix_sync(frag_lhs[1], &LHS(wy + 16, i), 80);
      mma_sync(frag_accum[1][0], frag_lhs[1], frag_rhs[0], frag_accum[1][0]);
      mma_sync(frag_accum[1][1], frag_lhs[1], frag_rhs[1], frag_accum[1][1]);
    }
    __syncthreads();
  }
  store_matrix_sync(&OUT(wy, wx), frag_accum[0][0], 64, mem_row_major);
  store_matrix_sync(&OUT(wy, wx + 16), frag_accum[0][1], 64, mem_row_major);
  store_matrix_sync(&OUT(wy + 16, wx), frag_accum[1][0], 64, mem_row_major);
  store_matrix_sync(&OUT(wy + 16, wx + 16), frag_accum[1][1], 64, mem_row_major);
  __syncthreads();
  int tx = tlo * 4, ty = thi * 8;
  int r = by + ty, c = bx + tx;
  for (int k = 0; k < 8; ++k) {
    for (int j = 0; j < 4; ++j) {
      int out_idx = (r + k) * n + c + j;
      output[out_idx] = OUT(ty + k, tx + j) + beta * __half2float(output[out_idx]);
    }
  }
#undef SDATA
#undef LHS
#undef RHS
#undef OUT
}

// output = lhs * rhs^T; lhs = (m, p); rhs = (n, p); output = (m, n)
// blockDim must be (256, 1, 1). gridDim must be (m/16 * n/128, 1, 1)
// m must be a multiple of 16, n must be a multiple of 128 and p a multiple of 256
__global__ void matmul_nt_wmma_16x128x256(__half* __restrict__ output,
                                          __half const* __restrict__ lhs,
                                          __half const* __restrict__ rhs,
                                          int m,
                                          int p,
                                          int n,
                                          float beta = 0.0f) {
  using namespace nvcuda::wmma;
  extern __shared__ void* sdata[];
  const int SDATA_BASE_LHS = 0;
  const int SDATA_BASE_RHS = 16 * 272;
#define SDATA(type, side, stride, d0, d1) \
  (((type*)sdata)[SDATA_BASE_##side + ((d0) * (stride)) + (d1)])
#define LHS(d0, d1) SDATA(__half, LHS, 272, d0, d1)
#define RHS(d0, d1) SDATA(__half, RHS, 272, d0, d1)
#define OUT(d0, d1) SDATA(float, LHS, 128, d0, d1)
  int bid = blockIdx.x;
  int dim_y = m / 16;
  int bx = (bid / dim_y) * 128, by = (bid % dim_y) * 16;
  unsigned tid = threadIdx.x;
  int tlo = tid & 63, thi = tid >> 6;
  int warp_id = tid / 32;
  int wx = 32 * (warp_id >> 1);
  fragment<accumulator, 16, 16, 16, float> frag_accum[2];
  fragment<matrix_a, 16, 16, 16, __half, row_major> frag_lhs;
  fragment<matrix_b, 16, 16, 16, __half, col_major> frag_rhs[2];
  fill_fragment(frag_accum[0], 0.0f);
  fill_fragment(frag_accum[1], 0.0f);
  for (int t = 0; t < p; t += 256) {
    for (int i = 0; i < 4; ++i) {
      int lhs_idx = (by + thi * 4 + i) * p + t + tlo * 4;
      *((short4*)&LHS(thi * 4 + i, tlo * 4)) = *(short4*)(&lhs[lhs_idx]);
    }
    for (int i = 0; i < 32; ++i) {
      int rhs_idx = (bx + thi * 32 + i) * p + t + tlo * 4;
      *((short4*)&RHS(thi * 32 + i, tlo * 4)) = *(short4*)(&rhs[rhs_idx]);
    }
    __syncthreads();
    for (int i = 0; i < 256; i += 16) {
      load_matrix_sync(frag_lhs, &LHS(0, i), 272);
      load_matrix_sync(frag_rhs[0], &RHS(wx, i), 272);
      mma_sync(frag_accum[0], frag_lhs, frag_rhs[0], frag_accum[0]);
      load_matrix_sync(frag_rhs[1], &RHS(wx + 16, i), 272);
      mma_sync(frag_accum[1], frag_lhs, frag_rhs[1], frag_accum[1]);
    }
    __syncthreads();
  }
  store_matrix_sync(&OUT(0, wx), frag_accum[0], 128, mem_row_major);
  store_matrix_sync(&OUT(0, wx + 16), frag_accum[1], 128, mem_row_major);
  __syncthreads();
  int tx = (tid & 15) * 8, ty = (tid >> 4);
  int r = by + ty, c = bx + tx;
  for (int j = 0; j < 8; ++j) {
    int out_idx = r * n + c + j;
    output[out_idx] = OUT(ty, tx + j) + beta * __half2float(output[out_idx]);
  }
#undef SDATA
#undef LHS
#undef RHS
#undef OUT
}

__device__ __forceinline__ half dequantize_absmax_one(uint8_t x, half scale) {
  return (float(x) - 127.5f) * __half2float(scale);
}

// output = lhs * rhs^T; lhs = (m, p); rhs = (n, p); output = (m, n)
// blockDim must be (256, 1, 1). gridDim must be (m/16 * n/128, 1, 1)
// m must be a multiple of 16, n must be a multiple of 128 and p a multiple of 256
__global__ void matmul_nt_wmma_16x128x256_fp16u8(half* __restrict__ output,
                                                 half const* __restrict__ lhs,
                                                 uint8_t const* __restrict__ rhs,
                                                 half const* __restrict__ rhs_scale,
                                                 int m,
                                                 int p,
                                                 int n,
                                                 int block_size,
                                                 float beta = 0.0f) {
  using namespace nvcuda::wmma;
  extern __shared__ void* sdata[];
  const int SDATA_BASE_LHS = 0;
  const int SDATA_BASE_RHS = 16 * 272;
#define SDATA(type, side, stride, d0, d1) \
  (((type*)sdata)[SDATA_BASE_##side + ((d0) * (stride)) + (d1)])
#define LHS(d0, d1) SDATA(__half, LHS, 272, d0, d1)
#define RHS(d0, d1) SDATA(__half, RHS, 272, d0, d1)
#define OUT(d0, d1) SDATA(float, LHS, 128, d0, d1)
  int bid = blockIdx.x;
  int dim_y = m / 16;
  int bx = (bid / dim_y) * 128, by = (bid % dim_y) * 16;
  unsigned tid = threadIdx.x;
  int tlo = tid & 63, thi = tid >> 6;
  int warp_id = tid / 32;
  int wx = 32 * (warp_id >> 1);
  int block_count = p / block_size;
  fragment<accumulator, 16, 16, 16, float> frag_accum[2];
  fragment<matrix_a, 16, 16, 16, __half, row_major> frag_lhs;
  fragment<matrix_b, 16, 16, 16, __half, col_major> frag_rhs[2];
  fill_fragment(frag_accum[0], 0.0f);
  fill_fragment(frag_accum[1], 0.0f);
  for (int t = 0; t < p; t += 256) {
    for (int i = 0; i < 4; ++i) {
      int lhs_idx = (by + thi * 4 + i) * p + t + tlo * 4;
      *((short4*)&LHS(thi * 4 + i, tlo * 4)) = *(short4*)(&lhs[lhs_idx]);
    }
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      half scale =
          __ldg(&rhs_scale[(bx + thi * 32 + i) * block_count + (t + tlo * 4) / block_size]);
      int rhs_idx = (bx + thi * 32 + i) * p + t + tlo * 4;
      uint32_t rhs_unscaled = *((uint32_t*)&rhs[rhs_idx]);
      half rhs_scaled[4];
      for (int j = 0; j < 4; ++j) {
        rhs_scaled[j] = dequantize_absmax_one(rhs_unscaled & 0xFF, scale);
        rhs_unscaled >>= 8;
      }
      *((short4*)&RHS(thi * 32 + i, tlo * 4)) = *(short4*)rhs_scaled;
    }
    __syncthreads();
    for (int i = 0; i < 256; i += 16) {
      load_matrix_sync(frag_lhs, &LHS(0, i), 272);
      load_matrix_sync(frag_rhs[0], &RHS(wx, i), 272);
      mma_sync(frag_accum[0], frag_lhs, frag_rhs[0], frag_accum[0]);
      load_matrix_sync(frag_rhs[1], &RHS(wx + 16, i), 272);
      mma_sync(frag_accum[1], frag_lhs, frag_rhs[1], frag_accum[1]);
    }
    __syncthreads();
  }
  store_matrix_sync(&OUT(0, wx), frag_accum[0], 128, mem_row_major);
  store_matrix_sync(&OUT(0, wx + 16), frag_accum[1], 128, mem_row_major);
  __syncthreads();
  int tx = (tid & 15) * 8, ty = (tid >> 4);
  int r = by + ty, c = bx + tx;
  for (int j = 0; j < 8; ++j) {
    int out_idx = r * n + c + j;
    output[out_idx] = OUT(ty, tx + j) + beta * __half2float(output[out_idx]);
  }
#undef SDATA
#undef LHS
#undef RHS
#undef OUT
}

__global__ void rotary(__half* output,
                       __half* input,
                       int h,
                       int w,
                       int n_heads,
                       int pos_offset = 0,
                       float theta = 10000.0) {
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  int head_dim = w / n_heads;
  int head_c = c % head_dim;

  if (r < h && c < w) {
    float angle = (pos_offset + r) / powf(theta, float(head_c) / head_dim);
    float real = __half2float(input[r * w + c]);
    float imag = __half2float(input[r * w + c + 1]);
    float a_cos = cosf(angle);
    float a_sin = sinf(angle);
    output[r * w + c] = __float2half(real * a_cos - imag * a_sin);
    output[r * w + c + 1] = __float2half(real * a_sin + imag * a_cos);
  }
}

// q * k^T
// output = (n_heads, seq_len_new, seq_len)
// lhs = (seq_len_new, dim)
// rhs = (seq_len, dim)
// we have to do the swizzle to regroup by heads here
// basically we have the above as inputs but we want to access them like
// lhs = (n_heads, seq_len_new, head_dim)
// rhs = (n_heads, head_dim, seq_len)
// grid x/y are row/col indices for the output, z is the head index
__global__ void matmul_qk(__half* output,
                          __half* lhs,
                          __half* rhs,
                          int seq_len_new,
                          int seq_len,
                          int dim,
                          int n_heads,
                          int start_pos) {
  // TODO: write a tiled kernel for this. only for testing accuracy.
  // probably write a cuBLAS path too which will need to materialize the
  // transposes.
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int head = blockIdx.z;
  int head_dim = dim / n_heads;
  bool masked = c > (r + start_pos);
  if (r < seq_len_new && c < seq_len) {
    if (masked) {
      output[head * seq_len_new * seq_len + r * seq_len_new + c] = -CUDART_INF_F;
    } else {
      float sum = 0;
      for (int i = 0; i < head_dim; i++) {
        sum += __half2float(lhs[r * dim + head * head_dim + i]) *
               __half2float(rhs[c * dim + head * head_dim + i]);
      }
      output[head * seq_len_new * seq_len + r * seq_len_new + c] = sum / sqrt(float(head_dim));
    }
  }
}

// row-wise softmax with optional temperature
// input = (n_heads, seq_len, seq_len)
// output = (n_heads, seq_len, seq_len)
__global__ void softmax_rows(__half* output, __half* input, int h, int w, float temp = 1.0) {
  int row = blockIdx.x;
  int head = blockIdx.y;
  int tid = threadIdx.x;
  int row_idx = head * h * w + row * w;
  bool warp_leader = tid % 32 == 0;
  int warp_id = tid / 32;
  __shared__ float s_max_val, s_sum_exp;
  __shared__ float s_warp_reduced[8];
  // max: thread reduction
  float max_val = -CUDART_INF_F;
  for (int i = tid; i < w; i += blockDim.x) {
    max_val = fmaxf(max_val, __half2float(input[row_idx + i]) / temp);
  }
  __syncthreads();
  // max: warp reduction
  for (int offset = 16; offset > 0; offset /= 2) {
    float other_val = __shfl_xor_sync(~0, max_val, offset);
    max_val = fmaxf(max_val, other_val);
  }
  if (warp_leader) {
    s_warp_reduced[warp_id] = max_val;
  }
  // max: block reduction
  __syncthreads();
  if (warp_id == 0) {
    max_val = (tid < 8) ? s_warp_reduced[tid] : -CUDART_INF_F;
    for (int offset = 4; offset > 0; offset /= 2) {
      float other_val = __shfl_xor_sync(~0, max_val, offset);
      max_val = fmaxf(max_val, other_val);
    }
    if (warp_leader) {
      s_max_val = max_val;
    }
  }
  __syncthreads();
  float sum_val = 0.0f;
  // expsum: thread reduction
  for (int i = tid; i < w; i += blockDim.x) {
    float val = __half2float(input[row_idx + i]) / temp;
    sum_val += expf(val - s_max_val);
  }
  __syncthreads();
  // expsum: warp reduction
  for (int offset = 16; offset > 0; offset /= 2) {
    sum_val += __shfl_xor_sync(~0, sum_val, offset);
  }
  if (warp_leader) {
    s_warp_reduced[warp_id] = sum_val;
  }
  // expsum: block reduction
  __syncthreads();
  if (warp_id == 0) {
    sum_val = (tid < 8) ? s_warp_reduced[tid] : 0.0f;
    for (int offset = 4; offset > 0; offset /= 2) {
      sum_val += __shfl_xor_sync(~0, sum_val, offset);
    }
    if (warp_leader) {
      s_sum_exp = sum_val;
    }
  }
  __syncthreads();
  for (int i = tid; i < w; i += blockDim.x) {
    float val = __half2float(input[row_idx + i]) / temp;
    output[row_idx + i] = __float2half(expf(val - s_max_val) / s_sum_exp);
  }
}

// lhs: (n_heads, seq_len_new, seq_len)
// rhs: (seq_len, dim)
// output: (seq_len_new, dim)
// logically we are viewing rhs as (seq_len, n_heads, head_dim) and then swizzling to
// (n_heads, seq_len, head_dim)
// then (n_heads, seq_len_new, seq_len) * (n_heads, seq_len, head_dim) ==>
// transpose (n_heads, seq_len_new, head_dim) ==>
// view (seq_len_new, n_heads, head_dim) ==>
// (seq_len_new, dim)
// but we're going to do all that without actually doing the transposes!
__global__ void matmul_qkv(__half* output,
                           __half* lhs,
                           __half* rhs,
                           int seq_len_new,
                           int seq_len,
                           int dim,
                           int n_heads) {
  // TODO: write a tiled version of this.
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int head_dim = dim / n_heads;
  int head = c / head_dim;
  if (r < seq_len_new && c < dim) {
    float sum = 0.0f;
    for (int i = 0; i < seq_len; i++) {
      sum += __half2float(lhs[head * seq_len_new * seq_len + r * seq_len + i]) *
             __half2float(rhs[i * dim + c]);
    }
    output[r * dim + c] = __float2half(sum);
  }
}

__global__ void silu(__half* output, __half* lhs, __half* rhs, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float val = __half2float(lhs[idx]);
    output[idx] = __float2half(val / (1.0f + expf(-val)) * __half2float(rhs[idx]));
  }
}
}  // namespace kernel

mapped_buffer::mapped_buffer(std::string const& path) {
#if defined(_WIN32)
  // TODO(eiz): support things other than america letters
  fd_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                    FILE_ATTRIBUTE_NORMAL, nullptr);
  if (fd_ == INVALID_HANDLE_VALUE) {
    return;
  }
  size_ = GetFileSize(fd_, nullptr);
  mapping_ = CreateFileMapping(fd_, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (mapping_ == nullptr) {
    CloseHandle(fd_);
    fd_ = INVALID_HANDLE_VALUE;
    return;
  }
  ptr_ = MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0);
#else
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    return;
  }
  fd_ = fd;
  size_ = lseek(fd_, 0, SEEK_END);
  ptr_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (ptr_ == MAP_FAILED) {
    ptr_ = nullptr;
  }
#endif
}
mapped_buffer::mapped_buffer(std::string const& path, size_t size) {
#if defined(_WIN32)
  fd_ = CreateFileA(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                    FILE_ATTRIBUTE_NORMAL, nullptr);
  if (fd_ == INVALID_HANDLE_VALUE) {
    return;
  }
  size_ = size;
  mapping_ = CreateFileMapping(fd_, nullptr, PAGE_READWRITE, 0, size, nullptr);
  if (mapping_ == nullptr) {
    CloseHandle(fd_);
    fd_ = INVALID_HANDLE_VALUE;
    return;
  }
  ptr_ = MapViewOfFile(mapping_, FILE_MAP_ALL_ACCESS, 0, 0, 0);
#else
  int fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    return;
  }
  fd_ = fd;
  size_ = size;
  if (ftruncate(fd_, size_) < 0) {
    close(fd_);
    fd_ = -1;
    return;
  }
  ptr_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  if (ptr_ == MAP_FAILED) {
    ptr_ = nullptr;
  }
#endif
}
mapped_buffer::mapped_buffer(size_t size) {
#if defined(_WIN32)
  mapping_ = CreateFileMapping(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, size, nullptr);
  if (mapping_ == nullptr) {
    return;
  }
  size_ = size;
  ptr_ = MapViewOfFile(mapping_, FILE_MAP_ALL_ACCESS, 0, 0, 0);
#else
  size_ = size;
  fd_ = -1;
  ptr_ = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr_ == MAP_FAILED) {
    ptr_ = nullptr;
  }
#endif
}
mapped_buffer::~mapped_buffer() {
#if defined(_WIN32)
  if (ptr_) {
    UnmapViewOfFile(ptr_);
  }
  if (mapping_) {
    CloseHandle(mapping_);
  }
  if (fd_ != INVALID_HANDLE_VALUE) {
    CloseHandle(fd_);
  }
#else
  if (ptr_) {
    munmap(ptr_, size_);
  }
  if (fd_ >= 0) {
    close(fd_);
  }
#endif
}
mapped_buffer::mapped_buffer(mapped_buffer&& other) {
  fd_ = other.fd_;
  ptr_ = other.ptr_;
  size_ = other.size_;
  other.fd_ = invalid_file;
  other.ptr_ = nullptr;
  other.size_ = 0;
#if defined(_WIN32)
  mapping_ = other.mapping_;
  other.mapping_ = nullptr;
#endif
}
mapped_buffer& mapped_buffer::operator=(mapped_buffer&& other) {
  if (this != &other) {
#if defined(_WIN32)
    if (ptr_) {
      UnmapViewOfFile(ptr_);
    }
    if (mapping_) {
      CloseHandle(mapping_);
    }
    if (fd_ != INVALID_HANDLE_VALUE) {
      CloseHandle(fd_);
    }
#else
    if (ptr_) {
      munmap(ptr_, size_);
    }
    if (fd_ >= 0) {
      close(fd_);
    }
#endif
    fd_ = other.fd_;
    ptr_ = other.ptr_;
    size_ = other.size_;
    other.fd_ = invalid_file;
    other.ptr_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

scoped_gpu::scoped_gpu(int device) {
  CHECK_CUDA(cudaGetDevice(&prev_device_));
  CHECK_CUDA(cudaSetDevice(device));
}
scoped_gpu::~scoped_gpu() {
  CHECK_CUDA(cudaSetDevice(prev_device_));
}

void gpu_buffer_view::copy_from(mapped_buffer const& buffer) {
  assert(buffer.size() == size_);
  CHECK_CUDA(cudaMemcpyAsync(dev_ptr_, buffer.data(), buffer.size(), cudaMemcpyHostToDevice));
}
void gpu_buffer_view::copy_from(void const* buffer, size_t size) {
  assert(size <= size_);
  CHECK_CUDA(cudaMemcpyAsync(dev_ptr_, buffer, size, cudaMemcpyHostToDevice));
}
void gpu_buffer_view::copy_to(void* buffer, size_t size) {
  assert(size <= size_);
  CHECK_CUDA(cudaMemcpy(buffer, dev_ptr_, size, cudaMemcpyDeviceToHost));
}

gpu_buffer::gpu_buffer(size_t size) {
  CHECK_CUDA(cudaMalloc(&dev_ptr_, size));
  size_ = size;
}
gpu_buffer::gpu_buffer(gpu_buffer&& other) : gpu_buffer_view(other.dev_ptr_, other.size_) {
  other.dev_ptr_ = nullptr;
  other.size_ = 0;
}
gpu_buffer& gpu_buffer::operator=(gpu_buffer&& other) {
  if (this != &other) {
    if (dev_ptr_) {
      CHECK_CUDA(cudaFree(dev_ptr_));
    }
    dev_ptr_ = other.dev_ptr_;
    size_ = other.size_;
    other.dev_ptr_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}
gpu_buffer::~gpu_buffer() {
  if (dev_ptr_) {
    CHECK_CUDA(cudaFree(dev_ptr_));
  }
}

void matmul_nt(tensor4d_view<__half> out,
               tensor4d_view<__half> lhs,
               tensor4d_view<__half> rhs,
               __half beta = 0.0f,
               cudaStream_t stream = 0);
void matmul_nt(tensor4d_view<__half> out,
               tensor4d_view<__half> lhs,
               generic_tensor& rhs,
               __half beta = 0.0f,
               cudaStream_t stream = 0);

constexpr int GUTTER = 128;

bool llama_params::load(mapped_buffer const& buffer) {
  if (buffer.size() != sizeof(llama_params)) {
    return false;
  }
  auto params = reinterpret_cast<llama_params const*>(buffer.data());
  if (params->magic != LLAMA_CU_MAGIC) {
    return false;
  }
  if (params->fourcc != LLAMA_CU_MODEL_FOURCC_LLAMA) {
    return false;
  }
  *this = *params;
  return true;
}
uint32_t llama_params::ffn_dim() const {
  return round_up(dim * 8 / 3, multiple_of);
}
size_t llama_params::layer_size() const {
  size_t norms = 2 * dim * sizeof(__half);
  size_t attn_weights, ffn_weights;
  if (q_type == quantization_type::fp16) {
    attn_weights = 4 * dim * dim * sizeof(__half);
    ffn_weights = 3 * ffn_dim() * dim * sizeof(__half);
  } else if (q_type == quantization_type::uint8) {
    attn_weights =
        4 * dim * dim * sizeof(uint8_t) + (dim / quantization_block_size * dim) * sizeof(__half);
    ffn_weights = 3 * ffn_dim() * dim * sizeof(uint8_t) +
                  (ffn_dim() / quantization_block_size * dim) * sizeof(__half) +
                  (dim / quantization_block_size * ffn_dim()) * sizeof(__half) * 2;
  } else {
    assert(false && "unsupported quantization type");
  }
  return attn_weights + norms + ffn_weights;
}
size_t llama_params::embedding_size() const {
  return n_vocab * dim * sizeof(__half);
}
size_t llama_params::output_size() const {
  return n_vocab * dim * sizeof(__half) + dim * sizeof(__half);
}
size_t llama_params::shared_context_memory_size(size_t context_len) const {
  size_t token_buf_size = context_len * sizeof(short);
  size_t embed_size = (context_len + GUTTER) * dim * sizeof(__half);
  size_t norm_size = embed_size;
  size_t xq_size = embed_size;
  size_t xk_size = embed_size;
  size_t qk_size = n_heads * (context_len + GUTTER) * context_len * sizeof(__half);
  size_t out_size = embed_size;
  size_t w1_size = (context_len + GUTTER) * ffn_dim() * sizeof(__half);
  size_t w3_size = w1_size;
  size_t silu_size = w1_size;
  size_t logits_size = (1 + GUTTER) * n_vocab * sizeof(__half);
  return token_buf_size + embed_size + norm_size + xq_size + xk_size + qk_size + out_size +
         w1_size + w3_size + silu_size + logits_size;
}
size_t llama_params::kv_cache_size(size_t context_len) const {
  return (context_len + GUTTER) * dim * 2 * sizeof(__half);
}

void llama_partition::use_default(llama_params const& params) {
  used_gpus = {0};
  embedding_gpu = 0;
  layer_gpus.clear();
  for (int i = 0; i < params.n_layers; ++i) {
    layer_gpus.push_back(0);
  }
  output_gpu = 0;
}
void llama_partition::autopipe(llama_params const& params, size_t context_len) {
  int cur_gpu = 0;
  size_t reserved_size = params.shared_context_memory_size(context_len) + 64 * 1024 * 1024;
  size_t cur_used = reserved_size;
  embedding_gpu = consume(cur_gpu, cur_used, reserved_size, params.embedding_size());
  for (int i = 0; i < params.n_layers; ++i) {
    layer_gpus.push_back(consume(cur_gpu, cur_used, reserved_size,
                                 params.layer_size() + params.kv_cache_size(context_len)));
  }
  output_gpu = consume(cur_gpu, cur_used, reserved_size, params.output_size());
  std::unordered_set<int> used_gpus_set;
  for (int i = 0; i < params.n_layers; ++i) {
    used_gpus_set.insert(layer_gpus[i]);
  }
  used_gpus_set.insert(embedding_gpu);
  used_gpus_set.insert(output_gpu);
  used_gpus.clear();
  used_gpus.insert(used_gpus.end(), used_gpus_set.begin(), used_gpus_set.end());
}
bool llama_partition::is_valid() {
  if (embedding_gpu == -1 || output_gpu == -1) {
    return false;
  }
  for (int i = 0; i < layer_gpus.size(); ++i) {
    if (layer_gpus[i] == -1) {
      return false;
    }
  }
  return true;
}
void llama_partition::debug_print() {
  fprintf(stderr, "Model partition:\n");
  fprintf(stderr, "  Embedding GPU: %d\n", embedding_gpu);
  fprintf(stderr, "  Output GPU: %d\n", output_gpu);
  fprintf(stderr, "  Layers:\n    ");
  for (int i = 0; i < layer_gpus.size(); i++) {
    fprintf(stderr, "% 3d: %d ", i, layer_gpus[i]);
    if ((i % 8) == 7 && i != layer_gpus.size() - 1) {
      fprintf(stderr, "\n    ");
    }
  }
  fprintf(stderr, "\n");
}
int llama_partition::consume(int& cur_gpu, size_t& cur_used, size_t reserved_size, size_t size) {
  if (cur_gpu == -1) {
    return -1;
  }
  int dev_count;
  CHECK_CUDA(cudaGetDeviceCount(&dev_count));
  do {
    size_t avail, total;
    scoped_gpu g(cur_gpu);
    CHECK_CUDA(cudaMemGetInfo(&avail, &total));
    if (avail - cur_used > size) {
      cur_used += size;
      return cur_gpu;
    }
    cur_gpu++;
    cur_used = reserved_size;
  } while (cur_gpu < dev_count);
  return -1;
}

struct llama_layer {
  generic_tensor wk, wq, wv, wo;
  generic_tensor w1, w2, w3;
  tensor4d<__half> ffn_norm, attn_norm;
};

struct llama_model_impl : public llama_model {
  llama_model_impl(llama_params params, llama_partition partition)
      : llama_model(params, partition) {}
  bool load_impl(std::string const& path) {
    {
      scoped_gpu g(partition.embedding_gpu);
      RETURN_UNLESS(
          load_tensor(tok_embeddings, path, "tok_embeddings", params.n_vocab, params.dim));
    }
    {
      scoped_gpu g(partition.output_gpu);
      RETURN_UNLESS(load_tensor(output, path, "output", params.n_vocab, params.dim));
      RETURN_UNLESS(load_tensor(norm, path, "norm", params.dim));
    }
    layers.clear();
    for (int i = 0; i < params.n_layers; ++i) {
      scoped_gpu g(partition.layer_gpus[i]);
      llama_layer layer;
      std::string basename = "layers." + std::to_string(i) + ".";
      RETURN_UNLESS(load_tensor(layer.attn_norm, path, basename + "attention_norm", params.dim));
      RETURN_UNLESS(load_tensor(layer.ffn_norm, path, basename + "ffn_norm", params.dim));
      RETURN_UNLESS(load_tensor(layer.wq, path, basename + "attention.wq", params.dim, params.dim));
      RETURN_UNLESS(load_tensor(layer.wk, path, basename + "attention.wk", params.dim, params.dim));
      RETURN_UNLESS(load_tensor(layer.wv, path, basename + "attention.wv", params.dim, params.dim));
      RETURN_UNLESS(load_tensor(layer.wo, path, basename + "attention.wo", params.dim, params.dim));
      RETURN_UNLESS(
          load_tensor(layer.w1, path, basename + "feed_forward.w1", params.ffn_dim(), params.dim));
      RETURN_UNLESS(
          load_tensor(layer.w2, path, basename + "feed_forward.w2", params.dim, params.ffn_dim()));
      RETURN_UNLESS(
          load_tensor(layer.w3, path, basename + "feed_forward.w3", params.ffn_dim(), params.dim));
      layers.emplace_back(std::move(layer));
    }
    return true;
  }
  generic_tensor tok_embeddings;
  tensor4d<__half> norm;
  generic_tensor output;
  std::vector<llama_layer> layers;

 private:
  bool load_tensor(generic_tensor& tensor,
                   std::string const& path,
                   std::string const& name,
                   size_t h,
                   size_t w) {
    if (params.q_type == quantization_type::fp16) {
      tensor4d<half> t_fp16(1, 1, h, w);
      mapped_buffer buf(path + "/" + name + ".weight__" + std::to_string(h) + "_" +
                        std::to_string(w));
      if (!buf.is_ok() || buf.size() != t_fp16.gpu.size()) {
        fprintf(stderr, "failed to load tensor %s (%zu x %zu)\n", name.c_str(), h, w);
        return false;
      }
      t_fp16.gpu.copy_from(buf);
      tensor = std::move(t_fp16);
      return true;
    } else if (params.q_type == quantization_type::uint8) {
      quantized_tensor quant;
      quant.q_values = tensor4d<uint8_t>(1, 1, h, w);
      quant.scales = tensor4d<half>(1, 1, h, w / params.quantization_block_size);
      quant.q_type = params.q_type;
      quant.quantization_block_size = params.quantization_block_size;
      mapped_buffer weight_buf(path + "/" + name + ".weight__" + std::to_string(h) + "_" +
                               std::to_string(w));
      if (!weight_buf.is_ok() || weight_buf.size() != quant.q_values.gpu.size()) {
        fprintf(stderr, "failed to load tensor %s (%zu x %zu)\n", name.c_str(), h, w);
        return false;
      }
      quant.q_values.gpu.copy_from(weight_buf);
      mapped_buffer scale_buf(path + "/" + name + ".scale__" + std::to_string(h) + "_" +
                              std::to_string(w / params.quantization_block_size));
      if (!scale_buf.is_ok() || scale_buf.size() != quant.scales.gpu.size()) {
        fprintf(stderr, "failed to load tensor scale %s (%zu x %zu)\n", name.c_str(), h,
                w / params.quantization_block_size);
        return false;
      }
      quant.scales.gpu.copy_from(scale_buf);
      tensor = std::move(quant);
      return true;
    }
    fprintf(stderr, "Unsupported quantization type %d\n", (int)params.q_type);
    return false;
  }
  bool load_tensor(tensor4d<half>& tensor,
                   std::string const& path,
                   std::string const& name,
                   size_t w) {
    tensor = tensor4d<half>(1, 1, 1, w);
    mapped_buffer buf(path + "/" + name + ".weight__" + std::to_string(w));
    if (!buf.is_ok() || buf.size() != tensor.gpu.size()) {
      fprintf(stderr, "failed to load tensor %s\n", name.c_str());
      return false;
    }
    tensor.gpu.copy_from(buf);
    return true;
  }
};

/* static */ std::unique_ptr<llama_model> llama_model::load(llama_params params,
                                                            llama_partition partition,
                                                            std::string const& path) {
  auto model = std::make_unique<llama_model_impl>(params, partition);

  if (!model->load_impl(path)) {
    return nullptr;
  }

  return model;
}

struct llama_context_impl final : public llama_context {
  llama_context_impl(llama_model* model, int context_len, float temperature)
      : llama_context(reinterpret_cast<llama_model_impl*>(model), context_len, temperature) {
    clear_tokens();
  }
  void clear_tokens() override {
    kv_layers_.clear();
    for (int i = 0; i < model_->params.n_layers; ++i) {
      scoped_gpu g(model_->partition.layer_gpus[i]);
      auto kv_layer = kv_cache_layer();
      kv_layer.k = tensor4d<__half>(1, 1, context_len_, model_->params.dim, GUTTER);
      kv_layer.v = tensor4d<__half>(1, 1, context_len_, model_->params.dim, GUTTER);
      kv_layers_.emplace_back(std::move(kv_layer));
    }
    gpu_contexts_.clear();
    for (auto gpu : model_->partition.used_gpus) {
      scoped_gpu g(gpu);
      auto gpu_context = per_gpu_context();
      gpu_context.t_embed = tensor4d<__half>(1, 1, context_len_, model_->params.dim, GUTTER);
      gpu_context.t_norm =
          tensor4d<__half>(1, 1, gpu_context.t_embed.h, gpu_context.t_embed.w, GUTTER);
      gpu_context.t_xq =
          tensor4d<__half>(1, 1, gpu_context.t_embed.h, gpu_context.t_embed.w, GUTTER);
      gpu_context.t_xk =
          tensor4d<__half>(1, 1, gpu_context.t_embed.h, gpu_context.t_embed.w, GUTTER);
      gpu_context.t_qk = tensor4d<__half>(1, model_->params.n_heads, gpu_context.t_embed.h,
                                          gpu_context.t_embed.h, GUTTER);
      gpu_context.t_out =
          tensor4d<__half>(1, 1, gpu_context.t_embed.h, gpu_context.t_embed.w, GUTTER);
      gpu_context.t_w1 =
          tensor4d<__half>(1, 1, gpu_context.t_norm.h, model_->params.ffn_dim(), GUTTER);
      gpu_context.t_w3 =
          tensor4d<__half>(1, 1, gpu_context.t_norm.h, model_->params.ffn_dim(), GUTTER);
      gpu_context.t_silu =
          tensor4d<__half>(1, 1, gpu_context.t_norm.h, model_->params.ffn_dim(), GUTTER);
      gpu_contexts_.emplace(gpu, std::move(gpu_context));
      CHECK_CUDA(cudaFuncSetAttribute(kernel::matmul_nt_wmma_16x128x256,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, 78336));
      CHECK_CUDA(cudaFuncSetAttribute(kernel::matmul_nt_wmma_16x128x256_fp16u8,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, 78336));
    }
    {
      scoped_gpu g(model_->partition.embedding_gpu);
      t_tokens_ = tensor4d<short>(1, 1, 1, context_len_);
    }
    {
      scoped_gpu g(model_->partition.output_gpu);
      t_logits_ = tensor4d<__half>(1, 1, 1, model_->params.n_vocab, GUTTER);
    }
    start_pos_ = 0;
  }
  size_t tokens_left() { return context_len_ - start_pos_; }
  size_t context_memory_usage() override {
    size_t count = 0;
    for (auto& layer : kv_layers_) {
      count += layer.k.gpu.size();
      count += layer.v.gpu.size();
    }
    for (auto& gpu_context : gpu_contexts_) {
      auto& ctx = gpu_context.second;
      count += ctx.t_embed.gpu.size();
      count += ctx.t_norm.gpu.size();
      count += ctx.t_xq.gpu.size();
      count += ctx.t_xk.gpu.size();
      count += ctx.t_qk.gpu.size();
      count += ctx.t_out.gpu.size();
      count += ctx.t_w1.gpu.size();
      count += ctx.t_w3.gpu.size();
      count += ctx.t_silu.gpu.size();
    }
    count += t_tokens_.gpu.size();
    count += t_logits_.gpu.size();
    return count;
  }
  short next_token(std::vector<short> const& new_tokens) override {
    assert(new_tokens.size() + start_pos_ <= context_len_);
    assert(gpu_contexts_.size() > 0);
    assert(model_->partition.embedding_gpu ==
           gpu_contexts_.find(*model_->partition.layer_gpus.begin())->first);
    assert(model_->partition.output_gpu ==
           gpu_contexts_.find(*model_->partition.layer_gpus.rbegin())->first);
    scoped_gpu embed_gpu(model_->partition.embedding_gpu);
    auto& embed_ctx = gpu_contexts_.at(model_->partition.embedding_gpu);
    auto& output_ctx = gpu_contexts_.at(model_->partition.output_gpu);
    auto v_tokens = t_tokens_.view().take(1, 1, 1, new_tokens.size());
    auto v_embed = embed_ctx.t_embed.view().take(1, 1, new_tokens.size(), model_->params.dim);
    auto v_logits = t_logits_.view().take(1, 1, 1, model_->params.n_vocab);
    v_tokens.gpu.copy_from(new_tokens.data(), new_tokens.size() * sizeof(short));
    if (model_->tok_embeddings.is_quantized()) {
      auto& t_quant = model_->tok_embeddings.as_quantized();
      // TODO: this is the exact sort of if statement that's going to cause trouble later.
      kernel::embed_uint8<<<ceil_div(new_tokens.size(), 256), 256>>>(
          model_->params.dim, new_tokens.size(), v_tokens.data(), v_embed.data(),
          t_quant.q_values.data(), t_quant.scales.data(), t_quant.quantization_block_size);
    } else {
      kernel::embed<<<ceil_div(new_tokens.size(), 256), 256>>>(
          model_->params.dim, new_tokens.size(), v_tokens.data(), v_embed.data(),
          model_->tok_embeddings.as_fp16().data());
    }
    CHECK_CUDA(cudaGetLastError());
    for (int cur_layer = 0; cur_layer < model_->layers.size(); ++cur_layer) {
      auto cur_layer_gpu_idx = model_->partition.layer_gpus[cur_layer];
      auto prev_layer_gpu_idx =
          cur_layer == 0 ? cur_layer_gpu_idx : model_->partition.layer_gpus[cur_layer - 1];
      auto& layer_ctx = gpu_contexts_.at(cur_layer_gpu_idx);
      auto& prev_layer_ctx = gpu_contexts_.at(prev_layer_gpu_idx);
      auto v_layer_embed =
          layer_ctx.t_embed.view().take(1, 1, new_tokens.size(), model_->params.dim);
      auto v_prev_layer_embed =
          prev_layer_ctx.t_embed.view().take(1, 1, new_tokens.size(), model_->params.dim);
      scoped_gpu layer_gpu(cur_layer_gpu_idx);
      if (cur_layer != 0 && prev_layer_gpu_idx != cur_layer_gpu_idx) {
        CHECK_CUDA(cudaMemcpyPeer(v_layer_embed.gpu.data(), cur_layer_gpu_idx,
                                  v_prev_layer_embed.gpu.data(), prev_layer_gpu_idx,
                                  v_layer_embed.gpu.size()));
      }
      auto v_norm = layer_ctx.t_norm.view().take(v_embed.n, v_embed.c, v_embed.h, v_embed.w);
      auto v_xq = layer_ctx.t_xq.view().take(v_embed.n, v_embed.c, v_embed.h, v_embed.w);
      auto v_xk =
          layer_ctx.t_xk.view().take(1, 1, new_tokens.size() + start_pos_, model_->params.dim);
      // TODO: matmul_qk/qkv need to handle strided access to make this a view
      auto tt_qk = tensor4d<__half>(1, model_->params.n_heads, new_tokens.size(),
                                    new_tokens.size() + start_pos_);
      auto v_qk = tt_qk.view();
      auto v_out = layer_ctx.t_out.view().take(1, 1, v_embed.h, v_embed.w);
      auto v_w1 = layer_ctx.t_w1.view().take(1, 1, v_norm.h, model_->params.ffn_dim());
      auto v_w3 = layer_ctx.t_w3.view().take(1, 1, v_norm.h, model_->params.ffn_dim());
      auto v_silu = layer_ctx.t_silu.view().take(1, 1, v_norm.h, model_->params.ffn_dim());
      dim3 gridDim;
      const dim3 blockDim(32, 32);
      kernel::rms_norm<<<v_embed.h, 256>>>(v_norm.data(), v_layer_embed.data(),
                                           model_->layers[cur_layer].attn_norm.data(), v_embed.h,
                                           v_embed.w, model_->params.norm_eps);
      CHECK_CUDA(cudaGetLastError());
      auto& kv_layer = kv_layers_[cur_layer];
      auto v_xk_full =
          kv_layer.k.view().take(1, 1, new_tokens.size() + start_pos_, model_->params.dim);
      auto v_xv_full =
          kv_layer.v.view().take(1, 1, new_tokens.size() + start_pos_, model_->params.dim);
      auto v_xk_new = v_xk_full.skip(0, 0, start_pos_, 0);
      auto v_xv_new = v_xv_full.skip(0, 0, start_pos_, 0);
      CHECK_CUDA(cudaEventRecord(layer_ctx.e_0, 0));
      CHECK_CUDA(cudaStreamWaitEvent(layer_ctx.s_v, layer_ctx.e_0));
      CHECK_CUDA(cudaStreamWaitEvent(layer_ctx.s_k, layer_ctx.e_0));
      // Q
      matmul_nt(v_xq, v_norm, model_->layers[cur_layer].wq);
      gridDim = dim3(ceil_div(v_xq.w, 32), ceil_div(v_xq.h, 32));
      kernel::rotary<<<gridDim, blockDim>>>(v_xq.data(), v_xq.data(), v_xq.h, v_xq.w,
                                            model_->params.n_heads, start_pos_);
      CHECK_CUDA(cudaGetLastError());
      // K
      matmul_nt(v_xk_new, v_norm, model_->layers[cur_layer].wk, 0.0, layer_ctx.s_k);
      gridDim = dim3(ceil_div(v_xk.w, 32), ceil_div(v_xk.h, 32));
      kernel::rotary<<<gridDim, blockDim, 0, layer_ctx.s_k>>>(
          v_xk.data(), v_xk_full.data(), v_xk_full.h, v_xk_full.w, model_->params.n_heads);
      CHECK_CUDA(cudaGetLastError());
      // V
      matmul_nt(v_xv_new, v_norm, model_->layers[cur_layer].wv, 0.0f, layer_ctx.s_v);
      CHECK_CUDA(cudaEventRecord(layer_ctx.e_k, layer_ctx.s_k));
      CHECK_CUDA(cudaStreamWaitEvent(0, layer_ctx.e_k));
      CHECK_CUDA(cudaEventRecord(layer_ctx.e_v, layer_ctx.s_v));
      CHECK_CUDA(cudaStreamWaitEvent(0, layer_ctx.e_v));
      gridDim = dim3(ceil_div(v_qk.w, 32), ceil_div(v_qk.h, 32), model_->params.n_heads);
      // softmax(QK^T / sqrt(head_dim)) * V
      kernel::matmul_qk<<<gridDim, blockDim>>>(v_qk.data(), v_xq.data(), v_xk.data(), v_xq.h,
                                               v_xk.h, v_xk.w, model_->params.n_heads, start_pos_);
      CHECK_CUDA(cudaGetLastError());
      kernel::softmax_rows<<<dim3(v_qk.h, model_->params.n_heads), 256>>>(v_qk.data(), v_qk.data(),
                                                                          v_qk.h, v_qk.w);
      CHECK_CUDA(cudaGetLastError());
      gridDim = dim3(ceil_div(v_xv_full.w, 32), ceil_div(v_xv_full.h, 32));
      kernel::matmul_qkv<<<gridDim, blockDim>>>(v_out.data(), v_qk.data(), v_xv_full.data(), v_qk.h,
                                                v_xv_full.h, v_xv_full.w, model_->params.n_heads);
      CHECK_CUDA(cudaGetLastError());
      // Residual
      matmul_nt(v_layer_embed, v_out, model_->layers[cur_layer].wo, 1.0f);
      // FFN
      kernel::rms_norm<<<v_embed.h, 256>>>(v_norm.data(), v_layer_embed.data(),
                                           model_->layers[cur_layer].ffn_norm.data(), v_embed.h,
                                           v_embed.w, model_->params.norm_eps);
      CHECK_CUDA(cudaGetLastError());
      matmul_nt(v_w1, v_norm, model_->layers[cur_layer].w1);
      matmul_nt(v_w3, v_norm, model_->layers[cur_layer].w3);
      gridDim = dim3(ceil_div(v_silu.w * v_silu.h, 256));
      kernel::silu<<<gridDim, 256>>>(v_silu.data(), v_w1.data(), v_w3.data(), v_silu.h * v_silu.w);
      CHECK_CUDA(cudaGetLastError());
      matmul_nt(v_layer_embed, v_silu, model_->layers[cur_layer].w2, 1.0f);
    }
    scoped_gpu output_gpu(model_->partition.output_gpu);
    auto v_output_norm = output_ctx.t_norm.view().take(v_embed.n, v_embed.c, v_embed.h, v_embed.w);
    auto v_output_embed =
        output_ctx.t_embed.view().take(v_embed.n, v_embed.c, v_embed.h, v_embed.w);
    kernel::rms_norm<<<v_embed.h, 256>>>(v_output_norm.data(), v_output_embed.data(),
                                         model_->norm.data(), v_embed.h, v_embed.w,
                                         model_->params.norm_eps);
    auto last_token_norm = v_output_norm.skip(0, 0, v_output_norm.h - 1, 0);
    matmul_nt(v_logits, last_token_norm, model_->output);
    kernel::softmax_rows<<<v_logits.h, 256>>>(v_logits.data(), v_logits.data(), v_logits.h,
                                              v_logits.w, temperature_);
    std::vector<__half> probs(v_logits.w);
    CHECK_CUDA(cudaMemcpy(probs.data(), v_logits.data(), v_logits.w * sizeof(__half),
                          cudaMemcpyDeviceToHost));
    start_pos_ += new_tokens.size();
    std::discrete_distribution<short> dist(probs.begin(), probs.end());
    return dist(rng_);

    // auto argmax = std::max_element(probs.begin(), probs.end(), [](__half lhs, __half rhs) {
    // return __half2float(lhs) < __half2float(rhs);
    //});

    // return std::distance(probs.begin(), argmax);
  }

 private:
  struct kv_cache_layer {
    tensor4d<__half> k;
    tensor4d<__half> v;
  };
  struct per_gpu_context {
    per_gpu_context() {
      CHECK_CUDA(cudaEventCreate(&e_0));
      CHECK_CUDA(cudaEventCreate(&e_k));
      CHECK_CUDA(cudaEventCreate(&e_v));
      CHECK_CUDA(cudaStreamCreateWithFlags(&s_k, cudaStreamNonBlocking));
      CHECK_CUDA(cudaStreamCreateWithFlags(&s_v, cudaStreamNonBlocking));
    }
    ~per_gpu_context() {
      if (e_0) {
        CHECK_CUDA(cudaEventDestroy(e_0));
      }
      if (e_k) {
        CHECK_CUDA(cudaEventDestroy(e_k));
      }
      if (e_v) {
        CHECK_CUDA(cudaEventDestroy(e_v));
      }
      if (s_k) {
        CHECK_CUDA(cudaStreamDestroy(s_k));
      }
      if (s_v) {
        CHECK_CUDA(cudaStreamDestroy(s_v));
      }
    }
    per_gpu_context(per_gpu_context&& other) {
      t_embed = std::move(other.t_embed);
      t_norm = std::move(other.t_norm);
      t_xq = std::move(other.t_xq);
      t_xk = std::move(other.t_xk);
      t_qk = std::move(other.t_qk);
      t_out = std::move(other.t_out);
      t_w1 = std::move(other.t_w1);
      t_w3 = std::move(other.t_w3);
      t_silu = std::move(other.t_silu);
      e_0 = other.e_0;
      e_k = other.e_k;
      e_v = other.e_v;
      s_k = other.s_k;
      s_v = other.s_v;
      other.e_0 = nullptr;
      other.e_k = nullptr;
      other.e_v = nullptr;
      other.s_k = nullptr;
      other.s_v = nullptr;
    }
    per_gpu_context& operator=(per_gpu_context&& other) {
      t_embed = std::move(other.t_embed);
      t_norm = std::move(other.t_norm);
      t_xq = std::move(other.t_xq);
      t_xk = std::move(other.t_xk);
      t_qk = std::move(other.t_qk);
      t_out = std::move(other.t_out);
      t_w1 = std::move(other.t_w1);
      t_w3 = std::move(other.t_w3);
      t_silu = std::move(other.t_silu);
      e_0 = other.e_0;
      e_k = other.e_k;
      e_v = other.e_v;
      s_k = other.s_k;
      s_v = other.s_v;
      other.e_0 = nullptr;
      other.e_k = nullptr;
      other.e_v = nullptr;
      other.s_k = nullptr;
      other.s_v = nullptr;
      return *this;
    }
    per_gpu_context(const per_gpu_context&) = delete;
    per_gpu_context& operator=(const per_gpu_context&) = delete;
    tensor4d<__half> t_embed;
    tensor4d<__half> t_norm;
    tensor4d<__half> t_xq;
    tensor4d<__half> t_xk;
    tensor4d<__half> t_qk;
    tensor4d<__half> t_out;
    tensor4d<__half> t_w1;
    tensor4d<__half> t_w3;
    tensor4d<__half> t_silu;
    cudaEvent_t e_0{}, e_k{}, e_v{};
    cudaStream_t s_k{}, s_v{};
  };
  std::vector<kv_cache_layer> kv_layers_;
  std::unordered_map<int, per_gpu_context> gpu_contexts_;
  tensor4d<short> t_tokens_;
  tensor4d<__half> t_logits_;
  std::random_device rng_;
};

/* static */ std::unique_ptr<llama_context> llama_context::create(llama_model* model,
                                                                  int context_len,
                                                                  float temperature) {
  return std::make_unique<llama_context_impl>(model, context_len, temperature);
}

static size_t utf8_len(char src) {
  const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
  uint8_t highbits = static_cast<uint8_t>(src) >> 4;
  return lookup[highbits];
}

llama_vocabulary::llama_vocabulary(mapped_buffer const& buffer) {
  struct entry {
    uint32_t offset;
    float score;
  };
  uint32_t num_entries = *reinterpret_cast<uint32_t*>(buffer.data());
  entry* entries = reinterpret_cast<entry*>(reinterpret_cast<uint32_t*>(buffer.data()) + 1);
  tokens.clear();
  tokens.reserve(num_entries);
  for (uint32_t i = 0; i < num_entries; ++i) {
    llama_token token;
    token.id = i;
    token.score = entries[i].score;
    if (i >= 3 && i < 259) {
      char c = (char)(i - 3);
      token.text = std::string(&c, 1);
    } else {
      token.text = std::string(reinterpret_cast<char*>(buffer.data()) + entries[i].offset);
    }
    tokens.push_back(token);
    if (i >= 259) {
      token_lookup.emplace(token.text, &tokens[i]);
    }
  }
}

std::vector<short> llama_vocabulary::tokenize(std::string_view text, bool bos) {
  llama_tokenizer tokenizer(*this);
  std::vector<short> output;
  if (bos) {
    output.push_back(1);
  }
  if (text.size() == 0) {
    return output;
  }
  if (text[0] != ' ') {
    std::string copy;
    copy.reserve(text.size() + 1);
    copy += ' ';
    copy += text;
    tokenizer.tokenize(copy, output);
  } else {
    tokenizer.tokenize(text, output);
  }
  return output;
}

void llama_tokenizer::tokenize(std::string_view text, std::vector<short>& output) {
  // split string into utf8 chars
  int index = 0;
  while (!text.empty()) {
    llama_sp_symbol sym;
    size_t char_len = std::min(text.size(), utf8_len(text.data()[0]));
    sym.text = std::string_view(text.data(), char_len);
    sym.prev = index - 1;
    text.remove_prefix(char_len);
    sym.next = text.empty() ? -1 : index + 1;
    index++;
    symbols_.emplace_back(std::move(sym));
  }
  // seed the work queue with all possible 2-character tokens.
  for (size_t i = 1; i < symbols_.size(); ++i) {
    try_add_bigram(i - 1, i);
  }
  // keep substituting the highest frequency pairs for as long as we can.
  while (!work_queue_.empty()) {
    auto bigram = work_queue_.top();
    work_queue_.pop();
    auto& left_sym = symbols_[bigram.left];
    auto& right_sym = symbols_[bigram.right];
    // if one of the symbols already got merged, skip it.
    if (left_sym.text.empty() || right_sym.text.empty() ||
        left_sym.text.size() + right_sym.text.size() != bigram.size) {
      continue;
    }
    // merge the right sym into the left one
    left_sym.text =
        std::string_view(left_sym.text.data(), left_sym.text.size() + right_sym.text.size());
    right_sym.text = std::string_view("");
    // remove the right sym from the chain
    left_sym.next = right_sym.next;
    if (right_sym.next >= 0) {
      symbols_[right_sym.next].prev = bigram.left;
    }
    // find more substitutions
    try_add_bigram(left_sym.prev, bigram.left);
    try_add_bigram(bigram.left, left_sym.next);
  }
  for (int i = 0; i != -1; i = symbols_[i].next) {
    auto& symbol = symbols_[i];
    auto token = vocab_.token_lookup.find(std::string(symbol.text));
    if (token == vocab_.token_lookup.end()) {
      // output any symbols that did not form tokens as bytes.
      for (int j = 0; j < symbol.text.size(); ++j) {
        short token_id = static_cast<uint8_t>(symbol.text[j]) + 3;
        output.push_back(token_id);
      }
    } else {
      output.push_back(token->second->id);
    }
  }
}

void llama_tokenizer::try_add_bigram(int left, int right) {
  if (left == -1 || right == -1) {
    return;
  }
  std::string_view text(symbols_[left].text.data(),
                        symbols_[left].text.size() + symbols_[right].text.size());
  auto token = vocab_.token_lookup.find(std::string(text));
  if (token == vocab_.token_lookup.end()) {
    return;
  }
  llama_sp_bigram bigram;
  bigram.left = left;
  bigram.right = right;
  bigram.score = token->second->score;
  bigram.size = text.size();
  work_queue_.push(bigram);
}

#ifdef ENABLE_CUBLAS
cublasHandle_t cublas_handle;

void matmul_nt_gemm(tensor4d_view<__half> out,
                    tensor4d_view<__half> lhs,
                    tensor4d_view<__half> rhs,
                    __half beta,
                    cudaStream_t stream) {
  int M = out.h, N = out.w, K = lhs.w;
  float fbeta = __half2float(beta);
  float falpha = 1.0f;
  cublasSetStream(cublas_handle, stream);
  // Because GEMM uses column major, we swap left/right and also swap T/N.
  cublasStatus_t status =
      cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &falpha, rhs.data(),
                   CUDA_R_16F, K, lhs.data(), CUDA_R_16F, K, &fbeta, out.data(), CUDA_R_16F, N,
                   CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP);
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("cuBLAS error %d\n", status);
    abort();
  }
}
#endif

void matmul_nt(tensor4d_view<__half> out,
               tensor4d_view<__half> lhs,
               tensor4d_view<__half> rhs,
               __half beta,
               cudaStream_t stream) {
  switch (matmul_type) {
#ifdef ENABLE_CUBLAS
    case 0:
      matmul_nt_gemm(out, lhs, rhs, beta, stream);
      break;
#endif
    case 1: {
      dim3 grid(ceil_div(rhs.h, 32), ceil_div(lhs.h, 32)), block(32, 32);
      kernel::matmul_nt<<<grid, block, 0, stream>>>(out.data(), lhs.data(), rhs.data(), lhs.h,
                                                    lhs.w, rhs.h, beta);
      break;
    }
    case 4: {
      if (lhs.h < 128) {
        dim3 grid(ceil_div(rhs.h, 128) * ceil_div(lhs.h, 16));
        kernel::matmul_nt_wmma_16x128x256<<<grid, 256, 78336, stream>>>(
            out.data(), lhs.data(), rhs.data(), round_up(lhs.h, 16), lhs.w, rhs.h, beta);
      } else {
        dim3 grid(ceil_div(rhs.h, 64) * ceil_div(lhs.h, 128));
        kernel::matmul_nt_wmma_128x64x64<<<grid, 256, 32768, stream>>>(
            out.data(), lhs.data(), rhs.data(), round_up(lhs.h, 128), lhs.w, rhs.h, beta);
      }
      break;
    }
    default:
      assert(false && "invalid matmul type");
  }
  CHECK_CUDA(cudaGetLastError());
}

void matmul_nt(tensor4d_view<__half> out,
               tensor4d_view<__half> lhs,
               generic_tensor& rhs,
               __half beta,
               cudaStream_t stream) {
  if (rhs.is_quantized()) {
    auto& t_quant = rhs.as_quantized();
    if (matmul_type == 4) {
      dim3 grid(ceil_div(t_quant.q_values.h, 128) * ceil_div(lhs.h, 16));
      kernel::matmul_nt_wmma_16x128x256_fp16u8<<<grid, 256, 78336, stream>>>(
          out.data(), lhs.data(), t_quant.q_values.data(), t_quant.scales.data(),
          round_up(lhs.h, 16), lhs.w, t_quant.q_values.h, t_quant.quantization_block_size, beta);
    } else if (matmul_type == 1) {
      dim3 grid(ceil_div(t_quant.q_values.h, 32), ceil_div(lhs.h, 32)), block(32, 32);
      kernel::matmul_nt_fp16u8<<<grid, block, 0, stream>>>(
          out.data(), lhs.data(), t_quant.q_values.data(), t_quant.scales.data(), lhs.h, lhs.w,
          t_quant.q_values.h, t_quant.quantization_block_size, beta);
    } else {
      assert(false && "invalid matmul type");
    }
    CHECK_CUDA(cudaGetLastError());
    // TODO: optimized kernels.
  } else {
    matmul_nt(out, lhs, rhs.as_fp16().view(), beta, stream);
  }
}

void initialize() {
  int device;
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  printf("Device name: %s\n", prop.name);
  printf("Number of SMs: %d\n", prop.multiProcessorCount);
  printf("Shared mem size per SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);
  printf("Shared mem size per block: %zu bytes\n", prop.sharedMemPerBlock);
  printf("Shared mem size per block (optin): %zu bytes\n", prop.sharedMemPerBlockOptin);
  printf("Shared mem size per block (reserved): %zu bytes\n", prop.reservedSharedMemPerBlock);
  printf("L2 cache size: %d bytes\n", prop.l2CacheSize);
#ifdef ENABLE_CUBLAS
  cublasCreate(&cublas_handle);
  cublasSetMathMode(cublas_handle, cublasMath_t(CUBLAS_DEFAULT_MATH |
                                                CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
#endif
}
}  // namespace llama_cu
