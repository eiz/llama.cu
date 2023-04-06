A standalone implementation of LLaMA LLM architecture in CUDA C++.

This doesn't really serve any purpose other than my personal learning about CUDA programming. I have not written any significant CUDA code prior to this, so a bunch of stuff (like the matmuls) is spelled out instead of using libraries for didactic reasons.

To build:

* Linux (WSL2 works fine)
* Have CUDA toolkit installed and in your PATH (I use 12.0 mainly)
* Have CMake >= 3.17 and Make or Ninja (if you're using Ninja, you know what to do different)
* `mkdir build && (cd build && cmake .. && make)`
* Download/extract [filthy_instruct_v6](https://f000.backblazeb2.com/file/unaligned-ai/filthy_instruct_v6_extracted.tar). It should extract to `models/filthy_instruct_v6` relative to the llama.cu directory. If you have access to the original LLaMA weights, you can go figure out how to extract them using `extract.py`. glhf
* `./build/llama_cu 4 filthy_instruct_v6 "My cool prompt"`
* If you want it to run a bit faster, define `ENABLE_CUBLAS` in `main.cu` and change the `4` to `0`

There's some big TODOs right now:

* make an actual CLI lol
* matmul_qk / matmul_qkv need tiled implementations (they are very slow) & cuBLAS equivalents with explicit transposes
* add quantization support (big refactor)
* multi-GPU for fp16 big models
* matmuls should use pipelining to load from global memory
* clean up extract.py / add support for common model file formats
* implement/test 13/30/65 models
* implement a more proper sampler
* create a proper library interface and tools
* it currently targets the RTX 4090. You can easily run it on Ampere devices by changing the architecture target, but anything less than 24GB or older than that will fail (either OOM or shared mem size limits). Probably will not bother fixing this until quantization is done.
* type in the 20 lines of code to port to Windows

The floating point calculations in here are as numerically close to the Meta pytorch implementation as I could get.

You can download [filthy_instruct_v6](https://f000.backblazeb2.com/file/unaligned-ai/filthy_instruct_v6_extracted.tar), my personal 7B test model. Don't say I didn't warn you. It is problematic af. The training set is a blend of old anime lemons, asstr, and Alpaca, all told about 160 million tokens. The instruct format is a bit different from Alpaca, try:

```
<|im_start|>system
You are a helpful assistant who follows instructions as accurately as possible.
<|im_start|>user
Write a story about ants where all the nouns are red.
<|im_start|>assistant
```

If you want to experience some very dark dreams, run it with an empty prompt ;)