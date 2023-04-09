#include <algorithm>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <fcntl.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef OUT
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#define ENABLE_CUBLAS
#ifdef ENABLE_CUBLAS
#include <cublas_v2.h>
#endif

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <mma.h>

#define CHECK_CUDA(expr)                                                                  \
  do {                                                                                    \
    cudaError_t __cuda_status = (expr);                                                   \
    if (__cuda_status != cudaSuccess) {                                                   \
      fprintf(stderr, "%s(%d): cuda error: %s (source: " #expr ")\n", __func__, __LINE__, \
              cudaGetErrorString(__cuda_status));                                         \
      abort();                                                                            \
    }                                                                                     \
  } while (0)
#define RETURN_UNLESS(x) \
  if (!(x)) {            \
    return false;        \
  }

using monoclock = std::chrono::steady_clock;

__device__ __host__ __forceinline__ size_t ceil_div(size_t a, size_t b) {
  return (a + b - 1) / b;
}

__device__ __host__ __forceinline__ size_t round_up(size_t a, size_t b) {
  return ceil_div(a, b) * b;
}

namespace kernel {
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
  __syncthreads();
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

// output = lhs * rhs^T; lhs = (m, p); rhs = (n, p); output = (m, n)
// blockDim must be (256, 1, 1). gridDim must be (m/128 * n/64, 1, 1)
// m must be a multiple of 128, n and p must be multiples of 64
__global__ void matmul_nt_wmma_128x64x64(__half* output,
                                         __half const* __restrict__ lhs,
                                         __half const* __restrict__ rhs,
                                         int m,
                                         int p,
                                         int n,
                                         float beta = 0.0f) {
  using namespace nvcuda::wmma;
  extern __shared__ __half sdata[];
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
__global__ void matmul_nt_wmma_16x128x256(__half* output,
                                          __half const* __restrict__ lhs,
                                          __half const* __restrict__ rhs,
                                          int m,
                                          int p,
                                          int n,
                                          float beta = 0.0f) {
  using namespace nvcuda::wmma;
  extern __shared__ __half sdata[];
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
  int tlo = tid & 15, thi = tid >> 4;
  int warp_id = tid / 32;
  int wx = 32 * (warp_id >> 1);
  fragment<accumulator, 16, 16, 16, float> frag_accum[2];
  fragment<matrix_a, 16, 16, 16, __half, row_major> frag_lhs;
  fragment<matrix_b, 16, 16, 16, __half, col_major> frag_rhs[2];
  fill_fragment(frag_accum[0], 0.0f);
  fill_fragment(frag_accum[1], 0.0f);
  for (int t = 0; t < p; t += 256) {
    for (int j = 0; j < 4; ++j) {
      int lhs_idx = (by + thi) * p + t + tlo * 16 + j * 4;
      *((short4*)&LHS(thi, tlo * 16 + j * 4)) = *(short4*)(&lhs[lhs_idx]);
    }

    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 4; ++j) {
        int rhs_idx = (bx + thi * 8 + i) * p + t + tlo * 16 + j * 4;
        *((short4*)&RHS(thi * 8 + i, tlo * 16 + j * 4)) = *(short4*)(&rhs[rhs_idx]);
      }
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
  int tx = tlo * 8, ty = thi;
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

struct mapped_buffer {
  mapped_buffer(std::string const& path) {
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
  mapped_buffer(size_t size) {
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
  ~mapped_buffer() {
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
  mapped_buffer(mapped_buffer&& other) {
    fd_ = other.fd_;
    ptr_ = other.ptr_;
    size_ = other.size_;
    mapping_ = other.mapping_;
    other.fd_ = invalid_file;
    other.ptr_ = nullptr;
    other.size_ = 0;
    other.mapping_ = nullptr;
  }
  mapped_buffer& operator=(mapped_buffer&& other) {
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
  bool is_ok() const { return ptr_ != nullptr; }
  void* data() const { return ptr_; }
  size_t size() const { return size_; }

 private:
  mapped_buffer(mapped_buffer const&) = delete;
  mapped_buffer& operator=(mapped_buffer const&) = delete;
#if defined(_WIN32)
  const HANDLE invalid_file = INVALID_HANDLE_VALUE;
  HANDLE fd_{invalid_file};
  HANDLE mapping_{nullptr};
#else
  const int invalid_file = -1;
  int fd_{invalid_file};
#endif
  void* ptr_{nullptr};
  size_t size_{0};
};

struct gpu_buffer_view {
  gpu_buffer_view() {}
  gpu_buffer_view(void* dev_ptr, size_t size) : dev_ptr_(dev_ptr), size_(size) {}
  void copy_from(mapped_buffer const& buffer) {
    assert(buffer.size() == size_);
    CHECK_CUDA(cudaMemcpyAsync(dev_ptr_, buffer.data(), buffer.size(), cudaMemcpyHostToDevice));
  }
  void copy_from(void const* buffer, size_t size) {
    assert(size <= size_);
    CHECK_CUDA(cudaMemcpyAsync(dev_ptr_, buffer, size, cudaMemcpyHostToDevice));
  }
  void copy_to(void* buffer, size_t size) {
    assert(size <= size_);
    CHECK_CUDA(cudaMemcpy(buffer, dev_ptr_, size, cudaMemcpyDeviceToHost));
  }
  bool is_ok() const { return dev_ptr_ != nullptr; }
  void* data() const { return dev_ptr_; }
  size_t size() const { return size_; }

 protected:
  void* dev_ptr_{nullptr};
  size_t size_{0};
};

struct gpu_buffer : public gpu_buffer_view {
  gpu_buffer() {}
  gpu_buffer(size_t size) {
    CHECK_CUDA(cudaMalloc(&dev_ptr_, size));
    size_ = size;
  }
  gpu_buffer(gpu_buffer&& other) : gpu_buffer_view(other.dev_ptr_, other.size_) {
    other.dev_ptr_ = nullptr;
    other.size_ = 0;
  }
  gpu_buffer& operator=(gpu_buffer&& other) {
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
  ~gpu_buffer() {
    if (dev_ptr_) {
      CHECK_CUDA(cudaFree(dev_ptr_));
    }
  }

 private:
  gpu_buffer(gpu_buffer const&) = delete;
  gpu_buffer& operator=(gpu_buffer const&) = delete;
};

template <typename T>
float to_float(T other_float);
template <>
float to_float(float other_float) {
  return other_float;
}
template <>
float to_float(__half other_float) {
  return __half2float(other_float);
}

template <typename T>
struct tensor4d;

template <typename T>
struct tensor4d_view {
  tensor4d_view() : n(0), c(0), h(0), w(0), stride_n(0), stride_c(0), stride_h(0) {}
  tensor4d_view(size_t n,
                size_t c,
                size_t h,
                size_t w,
                size_t stride_n,
                size_t stride_c,
                size_t stride_h,
                gpu_buffer_view gpu)
      : n(n),
        c(c),
        h(h),
        w(w),
        stride_n(stride_n),
        stride_c(stride_c),
        stride_h(stride_h),
        gpu(gpu) {}
  size_t size() { return n * c * h * w; }
  T* data() { return reinterpret_cast<T*>(gpu.data()); }
  tensor4d_view<T> skip(int skip_n, int skip_c, int skip_h, int skip_w) {
    tensor4d_view<T> result = *this;
    result.n -= skip_n;
    result.c -= skip_c;
    result.h -= skip_h;
    result.w -= skip_w;
    auto skip_el = skip_n * stride_n + skip_c * stride_c + skip_h * stride_h + skip_w;
    result.gpu = gpu_buffer_view(reinterpret_cast<char*>(result.gpu.data()) + skip_el * sizeof(T),
                                 gpu.size() - skip_el * sizeof(T));
    return result;
  }
  tensor4d_view<T> take(int take_n, int take_c, int take_h, int take_w) {
    tensor4d_view<T> result = *this;
    result.n = take_n;
    result.c = take_c;
    result.h = take_h;
    result.w = take_w;
    auto take_el = take_n * stride_n;
    result.gpu = gpu_buffer_view(result.gpu.data(), take_el * sizeof(T));
    return result;
  }
  void debug_print(char const* title, float divisor = 1.0) {
    T* cpu = (T*)malloc(gpu.size());
    gpu.copy_to(cpu, gpu.size());
    printf("==%s (%zu,%zu,%zu,%zu) (%zu,%zu,%zu)==\n", title, n, c, h, w, stride_n, stride_c,
           stride_h);
    for (int cur_c = 0; cur_c < c; ++cur_c) {
      printf("channel %d\n", cur_c);
      for (int cur_h = 0; cur_h < h; ++cur_h) {
        printf("% 4d ", cur_h);
        for (int cur_w = 0; cur_w < w; ++cur_w) {
          printf("% 9.6f, ", to_float(cpu[cur_c * stride_c + cur_h * stride_h + cur_w]) / divisor);
          if (cur_w > 1 && cur_w < w - 4 && w > 3) {
            printf(" ... ");
            cur_w = std::max<int>(cur_w, w - 4);
          }
        }
        printf("\n");
      }
      printf("\n\n\n");
    }
    free(cpu);
  }
  size_t n, c, h, w;
  size_t stride_n, stride_c, stride_h;
  gpu_buffer_view gpu;
};

template <typename T>
struct tensor4d {
  tensor4d() : n(0), c(0), h(0), w(0) {}
  // TODO(eiz): stupid gutter hack for 128x64 wmma kernel
  tensor4d(size_t n, size_t c, size_t h, size_t w, size_t gutter_h = 0)
      : n(n), c(c), h(h), w(w), gpu(gpu_buffer(n * c * (h + gutter_h) * w * sizeof(T))) {
    if (gutter_h > 0) {
      cudaMemset(gpu.data(), 0, gpu.size());
    }
  }
  size_t size() const { return n * c * h * w; }
  operator tensor4d_view<T>() { return tensor4d_view<T>(n, c, h, w, c * h * w, h * w, w, gpu); }
  tensor4d_view<T> view() { return *this; }
  T* data() { return reinterpret_cast<T*>(gpu.data()); }
  size_t n, c, h, w;
  gpu_buffer gpu;
};

void matmul_nt(tensor4d_view<__half> out,
               tensor4d_view<__half> lhs,
               tensor4d_view<__half> rhs,
               __half beta = 0.0f,
               cudaStream_t stream = 0);

struct llama_layer {
  tensor4d<__half> wk, wq, wv, wo;
  tensor4d<__half> w1, w2, w3;
  tensor4d<__half> ffn_norm, attn_norm;
};

struct llama_params {
  uint32_t dim;
  uint32_t multiple_of;
  uint32_t n_heads;
  uint32_t n_layers;
  uint32_t n_vocab;
  float norm_eps;
};

struct llama_model {
  llama_model(llama_params params) : params(params) {
    tok_embeddings = tensor4d<__half>(1, 1, params.n_vocab, params.dim);
    output = tensor4d<__half>(1, 1, params.n_vocab, params.dim);
    norm = tensor4d<__half>(1, 1, 1, params.dim);
    ffn_dim = params.multiple_of * ceil_div(params.dim * 8 / 3, params.multiple_of);
    for (int i = 0; i < params.n_layers; ++i) {
      layers.emplace_back();
      auto& layer = layers.back();
      layer.wk = tensor4d<__half>(1, 1, params.dim, params.dim);
      layer.wq = tensor4d<__half>(1, 1, params.dim, params.dim);
      layer.wv = tensor4d<__half>(1, 1, params.dim, params.dim);
      layer.wo = tensor4d<__half>(1, 1, params.dim, params.dim);
      layer.ffn_norm = tensor4d<__half>(1, 1, 1, params.dim);
      layer.attn_norm = tensor4d<__half>(1, 1, 1, params.dim);
      layer.w1 = tensor4d<__half>(1, 1, ffn_dim, params.dim);
      layer.w2 = tensor4d<__half>(1, 1, params.dim, ffn_dim);
      layer.w3 = tensor4d<__half>(1, 1, ffn_dim, params.dim);
    }
  }
  bool load(std::string const& path) {
    RETURN_UNLESS(load_tensor(tok_embeddings, path, "tok_embeddings", params.n_vocab, params.dim));
    RETURN_UNLESS(load_tensor(output, path, "output", params.n_vocab, params.dim));
    RETURN_UNLESS(load_tensor(norm, path, "norm", params.dim));
    for (int i = 0; i < params.n_layers; ++i) {
      auto& layer = layers[i];
      std::string basename = "layers." + std::to_string(i) + ".";
      RETURN_UNLESS(load_tensor(layer.attn_norm, path, basename + "attention_norm", params.dim));
      RETURN_UNLESS(load_tensor(layer.ffn_norm, path, basename + "ffn_norm", params.dim));
      RETURN_UNLESS(load_tensor(layer.wq, path, basename + "attention.wq", params.dim, params.dim));
      RETURN_UNLESS(load_tensor(layer.wk, path, basename + "attention.wk", params.dim, params.dim));
      RETURN_UNLESS(load_tensor(layer.wv, path, basename + "attention.wv", params.dim, params.dim));
      RETURN_UNLESS(load_tensor(layer.wo, path, basename + "attention.wo", params.dim, params.dim));
      RETURN_UNLESS(load_tensor(layer.w1, path, basename + "feed_forward.w1", ffn_dim, params.dim));
      RETURN_UNLESS(load_tensor(layer.w2, path, basename + "feed_forward.w2", params.dim, ffn_dim));
      RETURN_UNLESS(load_tensor(layer.w3, path, basename + "feed_forward.w3", ffn_dim, params.dim));
    }
    return true;
  }
  llama_params params;
  int ffn_dim;
  tensor4d<__half> tok_embeddings;
  tensor4d<__half> norm;
  tensor4d<__half> output;
  std::vector<llama_layer> layers;

 private:
  bool load_tensor(tensor4d<__half>& tensor,
                   std::string const& path,
                   std::string const& name,
                   size_t h,
                   size_t w) {
    mapped_buffer buf(path + "/" + name + ".weight__" + std::to_string(h) + "_" +
                      std::to_string(w));
    if (!buf.is_ok() || buf.size() != tensor.gpu.size()) {
      fprintf(stderr, "failed to load tensor %s\n", name.c_str());
      return false;
    }
    tensor.gpu.copy_from(buf);
    return true;
  }
  bool load_tensor(tensor4d<__half>& tensor,
                   std::string const& path,
                   std::string const& name,
                   size_t w) {
    mapped_buffer buf(path + "/" + name + ".weight__" + std::to_string(w));
    if (!buf.is_ok() || buf.size() != tensor.gpu.size()) {
      fprintf(stderr, "failed to load tensor %s\n", name.c_str());
      return false;
    }
    tensor.gpu.copy_from(buf);
    return true;
  }
};

struct llama_context {
  llama_context(llama_model* model, int context_len = 2048, float temperature = 0.9)
      : model_(model), context_len_(context_len), temperature_(temperature) {
    CHECK_CUDA(cudaEventCreate(&e_0_));
    CHECK_CUDA(cudaEventCreate(&e_k_));
    CHECK_CUDA(cudaEventCreate(&e_v_));
    CHECK_CUDA(cudaStreamCreate(&s_k_));
    CHECK_CUDA(cudaStreamCreate(&s_v_));
    reset();
  }
  void reset() {
    const int GUTTER = 128;
    kv_layers_.clear();
    for (int i = 0; i < model_->params.n_layers; ++i) {
      auto kv_layer = kv_cache_layer();
      kv_layer.k = tensor4d<__half>(1, 1, context_len_, model_->params.dim, GUTTER);
      kv_layer.v = tensor4d<__half>(1, 1, context_len_, model_->params.dim, GUTTER);
      kv_layers_.emplace_back(std::move(kv_layer));
    }
    t_tokens_ = tensor4d<short>(1, 1, 1, context_len_);
    t_embed_ = tensor4d<__half>(1, 1, context_len_, model_->params.dim, GUTTER);
    t_norm_ = tensor4d<__half>(1, 1, t_embed_.h, t_embed_.w, GUTTER);
    t_xq_ = tensor4d<__half>(1, 1, t_embed_.h, t_embed_.w, GUTTER);
    t_xk_ = tensor4d<__half>(1, 1, t_embed_.h, t_embed_.w, GUTTER);
    t_xv_ = tensor4d<__half>(1, 1, t_embed_.h, t_embed_.w, GUTTER);
    t_qk_ = tensor4d<__half>(1, model_->params.n_heads, t_embed_.h, t_embed_.h, GUTTER);
    t_out_ = tensor4d<__half>(1, 1, t_embed_.h, t_embed_.w, GUTTER);
    t_w1_ = tensor4d<__half>(1, 1, t_norm_.h, model_->ffn_dim, GUTTER);
    t_w3_ = tensor4d<__half>(1, 1, t_norm_.h, model_->ffn_dim, GUTTER);
    t_silu_ = tensor4d<__half>(1, 1, t_norm_.h, model_->ffn_dim, GUTTER);
    t_logits_ = tensor4d<__half>(1, 1, 1, model_->params.n_vocab, GUTTER);
    start_pos_ = 0;
  }
  size_t tokens_left() { return context_len_ - start_pos_; }
  size_t context_memory_usage() {
    size_t count = 0;

    for (auto& layer : kv_layers_) {
      count += layer.k.gpu.size();
      count += layer.v.gpu.size();
    }

    count += t_tokens_.gpu.size();
    count += t_embed_.gpu.size();
    count += t_norm_.gpu.size();
    count += t_xq_.gpu.size();
    count += t_xk_.gpu.size();
    count += t_xv_.gpu.size();
    count += t_qk_.gpu.size();
    count += t_out_.gpu.size();
    count += t_w1_.gpu.size();
    count += t_w3_.gpu.size();
    count += t_silu_.gpu.size();
    count += t_logits_.gpu.size();
    return count;
  }
  short next_token(std::vector<short> const& new_tokens) {
    assert(new_tokens.size() + start_pos_ <= context_len_);
    auto v_tokens = t_tokens_.view().take(1, 1, 1, new_tokens.size());
    auto v_embed = t_embed_.view().take(1, 1, new_tokens.size(), model_->params.dim);
    auto v_norm = t_norm_.view().take(v_embed.n, v_embed.c, v_embed.h, v_embed.w);
    auto v_xq = t_xq_.view().take(v_embed.n, v_embed.c, v_embed.h, v_embed.w);
    // TODO: matmul_qk/qkv need to handle strided access to make this a view
    auto tt_qk = tensor4d<__half>(1, model_->params.n_heads, new_tokens.size(),
                                  new_tokens.size() + start_pos_);
    auto v_qk = tt_qk.view();
    auto v_out = t_out_.view().take(1, 1, v_embed.h, v_embed.w);
    auto v_w1 = t_w1_.view().take(1, 1, v_norm.h, model_->ffn_dim);
    auto v_w3 = t_w3_.view().take(1, 1, v_norm.h, model_->ffn_dim);
    auto v_silu = t_silu_.view().take(1, 1, v_norm.h, model_->ffn_dim);
    auto v_logits = t_logits_.view().take(1, 1, 1, model_->params.n_vocab);
    v_tokens.gpu.copy_from(new_tokens.data(), new_tokens.size() * sizeof(short));
    kernel::embed<<<ceil_div(new_tokens.size(), 256), 256>>>(model_->params.dim, new_tokens.size(),
                                                             v_tokens.data(), v_embed.data(),
                                                             model_->tok_embeddings.data());
    for (int cur_layer = 0; cur_layer < model_->layers.size(); ++cur_layer) {
      dim3 gridDim;
      const dim3 blockDim(32, 32);
      kernel::rms_norm<<<v_embed.h, 256>>>(v_norm.data(), v_embed.data(),
                                           model_->layers[cur_layer].attn_norm.data(), v_embed.h,
                                           v_embed.w, model_->params.norm_eps);
      auto& kv_layer = kv_layers_[cur_layer];
      auto v_xk_full =
          kv_layer.k.view().take(1, 1, new_tokens.size() + start_pos_, model_->params.dim);
      auto v_xv_full =
          kv_layer.v.view().take(1, 1, new_tokens.size() + start_pos_, model_->params.dim);
      auto v_xk_new = v_xk_full.skip(0, 0, start_pos_, 0);
      auto v_xv_new = v_xv_full.skip(0, 0, start_pos_, 0);
      CHECK_CUDA(cudaEventRecord(e_0_, 0));
      CHECK_CUDA(cudaStreamWaitEvent(s_v_, e_0_));
      CHECK_CUDA(cudaStreamWaitEvent(s_k_, e_0_));
      // Q
      matmul_nt(v_xq, v_norm, model_->layers[cur_layer].wq);
      gridDim = dim3(ceil_div(v_xq.w, 32), ceil_div(v_xq.h, 32));
      kernel::rotary<<<gridDim, blockDim>>>(v_xq.data(), v_xq.data(), v_xq.h, v_xq.w,
                                            model_->params.n_heads, start_pos_);
      // K
      matmul_nt(v_xk_new, v_norm, model_->layers[cur_layer].wk, 0.0, s_k_);
      gridDim = dim3(ceil_div(v_xk_new.w, 32), ceil_div(v_xk_new.h, 32));
      kernel::rotary<<<gridDim, blockDim, 0, s_k_>>>(v_xk_new.data(), v_xk_new.data(), v_xk_new.h,
                                                     v_xk_new.w, model_->params.n_heads,
                                                     start_pos_);
      // V
      matmul_nt(v_xv_new, v_norm, model_->layers[cur_layer].wv, 0.0f, s_v_);
      CHECK_CUDA(cudaEventRecord(e_k_, s_k_));
      CHECK_CUDA(cudaStreamWaitEvent(0, e_k_));
      CHECK_CUDA(cudaEventRecord(e_v_, s_v_));
      CHECK_CUDA(cudaStreamWaitEvent(0, e_v_));
      gridDim = dim3(ceil_div(v_qk.w, 32), ceil_div(v_qk.h, 32), model_->params.n_heads);
      // softmax(QK^T / sqrt(head_dim)) * V
      kernel::matmul_qk<<<gridDim, blockDim>>>(v_qk.data(), v_xq.data(), v_xk_full.data(), v_xq.h,
                                               v_xk_full.h, v_xk_full.w, model_->params.n_heads,
                                               start_pos_);
      kernel::softmax_rows<<<dim3(v_qk.h, model_->params.n_heads), 256>>>(v_qk.data(), v_qk.data(),
                                                                          v_qk.h, v_qk.w);
      gridDim = dim3(ceil_div(v_xv_full.w, 32), ceil_div(v_xv_full.h, 32));
      kernel::matmul_qkv<<<gridDim, blockDim>>>(v_out.data(), v_qk.data(), v_xv_full.data(), v_qk.h,
                                                v_xv_full.h, v_xv_full.w, model_->params.n_heads);
      // Residual
      matmul_nt(v_embed, v_out, model_->layers[cur_layer].wo, 1.0f);
      // FFN
      kernel::rms_norm<<<v_embed.h, 256>>>(v_norm.data(), v_embed.data(),
                                           model_->layers[cur_layer].ffn_norm.data(), v_embed.h,
                                           v_embed.w, model_->params.norm_eps);
      matmul_nt(v_w1, v_norm, model_->layers[cur_layer].w1);
      matmul_nt(v_w3, v_norm, model_->layers[cur_layer].w3);
      gridDim = dim3(ceil_div(v_silu.w * v_silu.h, 256));
      kernel::silu<<<gridDim, 256>>>(v_silu.data(), v_w1.data(), v_w3.data(), v_silu.h * v_silu.w);
      matmul_nt(v_embed, v_silu, model_->layers[cur_layer].w2, 1.0f);
    }
    kernel::rms_norm<<<v_embed.h, 256>>>(v_norm.data(), v_embed.data(), model_->norm.data(),
                                         v_embed.h, v_embed.w, model_->params.norm_eps);
    auto last_token_norm = v_norm.skip(0, 0, v_norm.h - 1, 0);
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
  llama_model* model_;
  std::vector<kv_cache_layer> kv_layers_;
  int context_len_;
  int start_pos_{0};
  cudaEvent_t e_0_, e_k_, e_v_;
  cudaStream_t s_k_, s_v_;
  tensor4d<short> t_tokens_;
  tensor4d<__half> t_embed_;
  tensor4d<__half> t_norm_;
  tensor4d<__half> t_xq_;
  tensor4d<__half> t_xk_;
  tensor4d<__half> t_xv_;
  tensor4d<__half> t_qk_;
  tensor4d<__half> t_out_;
  tensor4d<__half> t_w1_;
  tensor4d<__half> t_w3_;
  tensor4d<__half> t_silu_;
  tensor4d<__half> t_logits_;
  std::random_device rng_;
  float temperature_;
};

struct llama_token {
  short id;
  float score;
  std::string text;
};

static size_t utf8_len(char src) {
  const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
  uint8_t highbits = static_cast<uint8_t>(src) >> 4;
  return lookup[highbits];
}

struct llama_sp_symbol {
  using index = int;
  index prev;
  index next;
  std::string_view text;
};

struct llama_sp_bigram {
  struct comparator {
    bool operator()(llama_sp_bigram& l, llama_sp_bigram& r) {
      return (l.score < r.score) || (l.score == r.score && l.left > r.left);
    }
  };
  using queue_storage = std::vector<llama_sp_bigram>;
  using queue = std::priority_queue<llama_sp_bigram, queue_storage, comparator>;
  llama_sp_symbol::index left;
  llama_sp_symbol::index right;
  float score;
  size_t size;
};

struct llama_vocabulary {
  llama_vocabulary(mapped_buffer const& buffer) {
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
  std::vector<short> tokenize(std::string_view text, bool bos);
  std::unordered_map<std::string, llama_token*> token_lookup;
  std::vector<llama_token> tokens;
};

struct llama_tokenizer {
  llama_tokenizer(llama_vocabulary const& vocab) : vocab_(vocab) {}
  void tokenize(std::string_view text, std::vector<short>& output);

 private:
  void try_add_bigram(int left, int right);
  llama_vocabulary const& vocab_;
  std::vector<llama_sp_symbol> symbols_;
  llama_sp_bigram::queue work_queue_;
};

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

int matmul_type = 4;

#ifdef ENABLE_CUBLAS
cublasHandle_t cublas_handle;

void matmul_nt_gemm(tensor4d_view<__half> out,
                    tensor4d_view<__half> lhs,
                    tensor4d_view<__half> rhs,
                    __half beta) {
  int M = out.h, N = out.w, K = lhs.w;
  float fbeta = __half2float(beta);
  float falpha = 1.0f;
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
      matmul_nt_gemm(out, lhs, rhs, beta);
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
}

int main(int argc, char const** argv) {
  int device;
  cudaDeviceProp prop;
  std::string model_name = "filthy_instruct_v6";
  std::string prompt_str = "Building a website can be done in 10 easy steps:";
  if (argc > 1) {
    matmul_type = atoi(argv[1]);
  }
  if (argc > 2) {
    model_name = argv[2];
  }
  if (argc > 3) {
    prompt_str = argv[3];
  }
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  printf("Device name: %s\n", prop.name);
  printf("Number of SMs: %d\n", prop.multiProcessorCount);
  printf("Shared mem size per SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);
  printf("Shared mem size per block: %zu bytes\n", prop.sharedMemPerBlock);
  printf("Shared mem size per block (optin): %zu bytes\n", prop.sharedMemPerBlockOptin);
  printf("Shared mem size per block (reserved): %zu bytes\n", prop.reservedSharedMemPerBlock);
  printf("L2 cache size: %d bytes\n", prop.l2CacheSize);
  CHECK_CUDA(cudaFuncSetAttribute(kernel::matmul_nt_wmma_128x64x64,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  prop.sharedMemPerBlockOptin));
  CHECK_CUDA(cudaFuncSetAttribute(kernel::matmul_nt_wmma_16x128x256,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  prop.sharedMemPerBlockOptin));
  mapped_buffer params_buf("models/" + model_name + "/params");
  mapped_buffer vocab_buf("models/" + model_name + "/vocab");
  if (!params_buf.is_ok()) {
    fprintf(stderr, "Could not load params file\n");
    return 1;
  }
  if (!vocab_buf.is_ok()) {
    fprintf(stderr, "Could not load vocab file\n");
    return 1;
  }
  llama_params params = *reinterpret_cast<llama_params*>(params_buf.data());
  llama_vocabulary vocab(vocab_buf);
  if (vocab.tokens.size() != params.n_vocab) {
    fprintf(stderr, "Vocab size mismatch\n");
    return 1;
  }
  llama_model model(params);
  if (!model.load("models/" + model_name)) {
    fprintf(stderr, "Could not load model\n");
    return 1;
  }
  printf(
      "Dimension: %d\nSwiGLU multiple: %d\nLayers: %d\nHeads: %d\nVocab: "
      "%d\nNorm Epsilon: %f\n",
      params.dim, params.multiple_of, params.n_layers, params.n_heads, params.n_vocab,
      params.norm_eps);
#ifdef ENABLE_CUBLAS
  cublasCreate(&cublas_handle);
  cublasSetMathMode(cublas_handle, cublasMath_t(CUBLAS_DEFAULT_MATH |
                                                CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
#endif
  llama_context context(&model, 2048, 0.8);
  printf("Context memory: %f MB\n", context.context_memory_usage() / 1024.0 / 1024.0);
#define BENCH
#ifdef BENCH
  std::vector<short> tokens;

  for (int i = 0; i < 2048; ++i) {
    tokens.push_back(1);
  }

  monoclock::time_point start, end;
  start = monoclock::now();
  context.next_token(tokens);
  end = monoclock::now();
  printf("Time: %f\n", (end - start).count() / 1e9);
#else
  std::vector<short> tokens = vocab.tokenize(prompt_str, true);
  auto prompt_tokens = tokens.size();
  struct timespec start, end;
  int tokens_generated = 0;
  std::vector<float> token_times;
  std::vector<short> complete_tokens = tokens;
  clock_gettime(CLOCK_MONOTONIC, &start);
  while (true) {
    struct timespec token_start, token_end;
    clock_gettime(CLOCK_MONOTONIC, &token_start);
    if (context.tokens_left() == 0) {
      context.reset();
      tokens.clear();
      tokens.insert(tokens.end(), complete_tokens.end() - 1024, complete_tokens.end());
    }
    auto next_token = context.next_token(tokens);
    clock_gettime(CLOCK_MONOTONIC, &token_end);
    if (next_token == 2) {
      break;
    }
    printf("%s", vocab.tokens[next_token].text.c_str());
    fflush(stdout);
    tokens.clear();
    tokens.push_back(next_token);
    complete_tokens.push_back(next_token);
    token_times.push_back((token_end.tv_sec - token_start.tv_sec) +
                          (token_end.tv_nsec - token_start.tv_nsec) / 1e9);
    tokens_generated++;
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  printf("\ntokens: %d time: %f\n", tokens_generated,
         (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);
  FILE* fp = fopen("times.txt", "w");
  assert(fp);
  fprintf(fp, "%zu\n", prompt_tokens);
  for (auto time : token_times) {
    fprintf(fp, "%f\n", time);
  }
  fclose(fp);
#endif
  return 0;
}