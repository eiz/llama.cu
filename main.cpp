#include <cassert>
#include <string>

#define LLAMA_CU_IMPLEMENTATION
#include "llama_cu.h"

int main(int argc, char const** argv) {
  size_t context_len = 2048;
  std::string model_name = "filthy_instruct_v6";
  std::string prompt_str = "Building a website can be done in 10 easy steps:";
  if (argc > 1) {
    llama_cu::matmul_type = atoi(argv[1]);
  }
  if (argc > 2) {
    model_name = argv[2];
  }
  if (argc > 3) {
    prompt_str = argv[3];
  }
  int device;

  llama_cu::initialize();
  llama_cu::mapped_buffer params_buf("models/" + model_name + "/params");
  llama_cu::mapped_buffer vocab_buf("models/" + model_name + "/vocab");
  if (!params_buf.is_ok()) {
    fprintf(stderr, "Could not load params file\n");
    return 1;
  }
  if (!vocab_buf.is_ok()) {
    fprintf(stderr, "Could not load vocab file\n");
    return 1;
  }
  llama_cu::llama_params params = *reinterpret_cast<llama_cu::llama_params*>(params_buf.data());
  llama_cu::llama_vocabulary vocab(vocab_buf);
  if (vocab.tokens.size() != params.n_vocab) {
    fprintf(stderr, "Vocab size mismatch\n");
    return 1;
  }
  llama_cu::llama_partition partition;
  partition.autopipe(params, context_len);
  partition.debug_print();
  if (!partition.is_valid()) {
    fprintf(stderr, "Could not partition model\n");
    return 1;
  }
  printf(
      "Dimension: %d\nSwiGLU multiple: %d\nLayers: %d\nHeads: %d\nVocab: "
      "%d\nNorm Epsilon: %f\nShared Context Memory: %f MB\nPer-Layer Context Memory: %f MB\n",
      params.dim, params.multiple_of, params.n_layers, params.n_heads, params.n_vocab,
      params.norm_eps, params.shared_context_memory_size(context_len) / 1024.0 / 1024.0,
      params.kv_cache_size(context_len) / 1024.0 / 1024.0);
  auto model = llama_cu::llama_model::load(params, partition, "models/" + model_name);
  if (!model) {
    fprintf(stderr, "Could not load model\n");
    return 1;
  }

  auto context = llama_cu::llama_context::create(&*model, context_len, 0.8);
  printf("Context memory: %f MB\n", context->context_memory_usage() / 1024.0 / 1024.0);
#ifdef BENCH
  std::vector<short> tokens;

  for (int i = 0; i < context_len; ++i) {
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
  llama_cu::monoclock::time_point start, end;
  int tokens_generated = 0;
  std::vector<float> token_times;
  std::vector<short> complete_tokens = tokens;
  start = llama_cu::monoclock::now();
  while (true) {
    llama_cu::monoclock::time_point token_start, token_end;
    token_start = llama_cu::monoclock::now();
    if (context->tokens_left() == 0) {
      context->clear_tokens();
      tokens.clear();
      tokens.insert(tokens.end(), complete_tokens.end() - context_len / 2, complete_tokens.end());
    }
    auto next_token = context->next_token(tokens);
    token_end = llama_cu::monoclock::now();
    if (next_token == 2) {
      break;
    }
    printf("%s", vocab.tokens[next_token].text.c_str());
    fflush(stdout);
    tokens.clear();
    tokens.push_back(next_token);
    complete_tokens.push_back(next_token);
    token_times.push_back((token_end - token_start).count() / 1e9);
    tokens_generated++;
  }
  end = llama_cu::monoclock::now();
  printf("\ntokens: %d time: %f\n", tokens_generated, (end - start).count() / 1e9);
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