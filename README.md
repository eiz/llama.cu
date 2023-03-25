A standalone implementation of LLaMA LLM architecture in CUDA C++.

This doesn't really serve any purpose other than my personal learning about CUDA programming. I have not written any significant CUDA code prior to this, so a bunch of stuff (like the matmuls) is spelled out instead of using libraries for didactic reasons.

There's some big TODOs right now:

* matmul_qk / matmul_qkv need tiled implementations (they are very slow) & cuBLAS equivalents with explicit transposes
* add quantization support (big refactor)
* matmuls should use pipelining to load from global memory
* clean up extract.py / add support for common model file formats
* implement/test 13/30/65 models
* implement a more proper sampler
* create a proper library interface and tools

The floating point calculations in here are as numerically close to the Meta pytorch implementation as I could get.

`filthy_instruct_v6` is not available at this time, sorry ;)