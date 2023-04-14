A standalone implementation of LLaMA LLM architecture in CUDA C++.

This doesn't really serve any purpose other than my personal learning about CUDA programming. I have not written any significant CUDA code prior to this, so a bunch of stuff (like the matmuls) is spelled out instead of using libraries for didactic reasons.

To build:

Linux:

* Have CUDA toolkit installed and in your PATH (I use 12.0 mainly)
* Have CMake >= 3.17 and Make or Ninja (if you're using Ninja, you know what to do different)
* `mkdir build && (cd build && cmake .. && make)`
* See below for sample model weights
* `./build/llama_cu 4 filthy_instruct_v6 "List of top 10 entertaining cat facts:"`

Windows:

* Install Visual Studio 2022 (Community is fine) + CUDA Toolkit + CMake >= 3.17
* Generate VS project with CMake and build.
* Models are loaded relative to the current working directory, so run it from the parent of your `models/` dir.

On either platform, you can speed things up by setting the `ENABLE_CUBLAS` CMake option.

Recent additions:

* Added pipeline parallel multi-GPU support. Can now load 30B in fp16 on my 4090/6000 box.

There's some big TODOs right now:

* make an actual CLI lol
* matmul_qk / matmul_qkv need tiled implementations (they are very slow) & cuBLAS equivalents with explicit transposes
* add quantization support (big refactor)
* matmuls should use pipelining to load from global memory
* clean up extract.py / add support for common model file formats
* implement/test 13/30/65 models
* implement a more proper sampler
* create a proper library interface and tools
* it currently targets the RTX 4090. You can easily run it on Ampere devices by changing the architecture target, but anything less than 24GB or older than that will fail (either OOM or shared mem size limits). Probably will not bother fixing this until quantization is done.

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
