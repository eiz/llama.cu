#pragma once

// clangd just isn't smart enough to figure this one out when editing llama_cu.h
#define LLAMA_CU_IMPLEMENTATION

// more clangd hacks see https://github.com/NVIDIA/thrust/issues/1703 some wacky stuff happens
#if defined(__noinline__) && defined(_CLANGD)
#undef __noinline__
#endif

#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#define LLAMA_CU_API __declspec(dllexport)
#else
// TODO ya ya ya symbol visibility w/e w/e
#define LLAMA_CU_API
#endif

#ifdef LLAMA_CU_IMPLEMENTATION
#include <chrono>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#endif  // LLAMA_CU_IMPLEMENTATION

namespace llama_cu {
#ifdef LLAMA_CU_IMPLEMENTATION
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

using monoclock = std::chrono::steady_clock;
#endif  // LLAMA_CU_IMPLEMENTATION

// lol, 0 is cublas, 1 is very pedantic slow implementation and 4 is my shitty wmma kernels
// 3 was "compare cublas to my kernels". rip
// set this before using the library. default is 4
extern LLAMA_CU_API int matmul_type;

struct mapped_buffer {
  mapped_buffer(std::string const& path);
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

enum class llama_quantization_type : uint8_t {
  fp16,
};

struct llama_params {
  // uint32_t magic;
  // uint32_t model_fourcc;
  // uint32_t struct_size;
  uint32_t dim;
  uint32_t multiple_of;
  uint32_t n_heads;
  uint32_t n_layers;
  uint32_t n_vocab;
  float norm_eps;
  // llama_quantization_type quantization_type;
  // uint8_t reserved0[3];
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