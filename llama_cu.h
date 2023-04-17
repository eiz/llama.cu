#pragma once

#if __cplusplus < 201703L
#error "C++17 or higher is required"
#endif

// clangd just isn't smart enough to figure this one out when editing llama_cu.h
// #define LLAMA_CU_IMPLEMENTATION

// more clangd hacks see https://github.com/NVIDIA/thrust/issues/1703 some wacky stuff happens
#if defined(__noinline__) && defined(_CLANGD)
#undef __noinline__
#endif

#include <memory>
#include <queue>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#define LLAMA_CU_API __declspec(dllexport)
#else
// TODO ya ya ya symbol visibility w/e w/e
#define LLAMA_CU_API
#endif

#ifdef LLAMA_CU_IMPLEMENTATION

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#endif

#include <cuda_fp16.h>

#define CHECK_CUDA(expr)                                                                  \
  do {                                                                                    \
    cudaError_t __cuda_status = (expr);                                                   \
    if (__cuda_status != cudaSuccess) {                                                   \
      fprintf(stderr, "%s(%d): cuda error: %s (source: " #expr ")\n", __func__, __LINE__, \
              cudaGetErrorString(__cuda_status));                                         \
      raise(SIGSEGV);                                                                     \
    }                                                                                     \
  } while (0)
#define RETURN_UNLESS(x) \
  if (!(x)) {            \
    return false;        \
  }

#endif  // LLAMA_CU_IMPLEMENTATION

namespace llama_cu {
// lol, 0 is cublas, 1 is very pedantic slow implementation and 4 is my shitty wmma kernels
// 3 was "compare cublas to my kernels". rip
// set this before using the library. default is 4
extern LLAMA_CU_API int matmul_type;

struct mapped_buffer {
  mapped_buffer(std::string const& path);
  mapped_buffer(std::string const& path, size_t size);
  mapped_buffer(size_t size);
  ~mapped_buffer();
  mapped_buffer(mapped_buffer&& other);
  mapped_buffer& operator=(mapped_buffer&& other);
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

struct scoped_gpu {
  scoped_gpu(int device);
  ~scoped_gpu();

 private:
  scoped_gpu(scoped_gpu const&) = delete;
  scoped_gpu& operator=(scoped_gpu const&) = delete;
  scoped_gpu(scoped_gpu&&) = delete;
  scoped_gpu& operator=(scoped_gpu&&) = delete;
  int prev_device_;
};

struct gpu_buffer_view {
  gpu_buffer_view() {}
  gpu_buffer_view(void* dev_ptr, size_t size) : dev_ptr_(dev_ptr), size_(size) {}
  void copy_from(mapped_buffer const& buffer);
  void copy_from(void const* buffer, size_t size);
  void copy_to(void* buffer, size_t size);
  bool is_ok() const { return dev_ptr_ != nullptr; }
  void* data() const { return dev_ptr_; }
  size_t size() const { return size_; }

 protected:
  void* dev_ptr_{nullptr};
  size_t size_{0};
};

struct gpu_buffer : public gpu_buffer_view {
  gpu_buffer() {}
  gpu_buffer(size_t size);
  gpu_buffer(gpu_buffer&& other);
  gpu_buffer& operator=(gpu_buffer&& other);
  ~gpu_buffer();

 private:
  gpu_buffer(gpu_buffer const&) = delete;
  gpu_buffer& operator=(gpu_buffer const&) = delete;
};

#ifdef LLAMA_CU_IMPLEMENTATION

template <typename T>
inline float to_float(T other_float);
template <>
inline float to_float(float other_float) {
  return other_float;
}
template <>
inline float to_float(__half other_float) {
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

#endif  // LLAMA_CU_IMPLEMENTATION

enum class llama_quantization_type : uint8_t {
  fp16,
  uint8,
};

#define LLAMA_CU_MAGIC 0x000041524F525541       // "AURORA\0\0" little endian
#define LLAMA_CU_MODEL_FOURCC_LLAMA 0x414D4C4C  // "LLMA" little endian

// There must be no padding between the fields of this struct.
// No fields may ever be removed or reordered.
// Your computer is assumed to be little endian.
struct llama_params {
  uint64_t magic;   // must be LLAMA_CU_MAGIC
  uint32_t fourcc;  // must be LLAMA_CU_MODEL_FOURCC_LLAMA
  uint32_t dim;
  uint32_t multiple_of;
  uint32_t n_heads;
  uint32_t n_layers;
  uint32_t n_vocab;
  float norm_eps;
  llama_quantization_type quantization_type;
  uint8_t reserved0[1];
  uint16_t quantization_block_size;
  bool load(mapped_buffer const& buffer);
  uint32_t ffn_dim() const;
  size_t layer_size() const;
  size_t embedding_size() const;
  size_t output_size() const;
  size_t shared_context_memory_size(size_t context_len) const;
  size_t kv_cache_size(size_t context_len) const;
};

struct llama_partition {
  llama_partition() {}
  void use_default(llama_params const& params);
  void autopipe(llama_params const& params, size_t context_len);
  bool is_valid();
  void debug_print();
  std::vector<int> used_gpus;
  int embedding_gpu{-1};
  std::vector<int> layer_gpus;
  int output_gpu{-1};

 private:
  int consume(int& cur_gpu, size_t& cur_used, size_t reserved_size, size_t size);
};

struct llama_token {
  short id;
  float score;
  std::string text;
};

struct llama_vocabulary {
  llama_vocabulary(mapped_buffer const& buffer);
  std::vector<short> tokenize(std::string_view text, bool bos);
  std::unordered_map<std::string, llama_token*> token_lookup;
  std::vector<llama_token> tokens;
};

struct llama_tokenizer {
  llama_tokenizer(llama_vocabulary const& vocab) : vocab_(vocab) {}
  void tokenize(std::string_view text, std::vector<short>& output);

 private:
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
  void try_add_bigram(int left, int right);
  llama_vocabulary const& vocab_;
  std::vector<llama_sp_symbol> symbols_;
  llama_sp_bigram::queue work_queue_;
};

struct llama_model_impl;

struct llama_model {
  static std::unique_ptr<llama_model> load(llama_params params,
                                           llama_partition partition,
                                           std::string const& path);
  llama_params params;
  llama_partition partition;

 protected:
  llama_model(llama_params params, llama_partition partition)
      : params(params), partition(partition) {}
};

struct llama_context {
  static std::unique_ptr<llama_context> create(llama_model* model,
                                               int context_len = 2048,
                                               float temperature = 0.9);
  virtual void clear_tokens() = 0;
  virtual size_t context_memory_usage() = 0;
  virtual short next_token(std::vector<short> const& new_tokens) = 0;
  size_t tokens_left() { return context_len_ - start_pos_; }

 protected:
  llama_context(llama_model_impl* model, int context_len, float temperature)
      : model_(model), context_len_(context_len), temperature_(temperature) {}
  llama_model_impl* model_;
  int context_len_;
  int start_pos_{0};
  float temperature_;
};

void initialize();

}  // namespace llama_cu