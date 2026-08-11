# Appendix A · Glossary

> 🌐 [中文版](../appendix-a-glossary) | English

> Appendix A | [Previous: Chapter 10 · Ecosystem Comparison](./10-ecosystem.md) | [Next: Appendix B · Struct Reference](./appendix-b-structs.md)

> This glossary organizes the terms that appear throughout the book (Chapters 1-10) by category.
> When you run into a word you don't understand while reading other chapters, look it up here.
> Every term is labeled with its **English / full name**, for easy cross-referencing with the HuggingFace docs and papers.

---

## Table of Contents

- [A.1 Model-Structure Terms](#a1-model-structure-terms)
- [A.2 Attention Terms](#a2-attention-terms)
- [A.3 Normalization & Activation Terms](#a3-normalization--activation-terms)
- [A.4 Tokenization Terms](#a4-tokenization-terms)
- [A.5 Data-Format Terms](#a5-data-format-terms)
- [A.6 C / Engineering Terms](#a6-c--engineering-terms)
- [A.7 Ecosystem & Performance Terms](#a7-ecosystem--performance-terms)

---

## A.1 Model-Structure Terms

| Term | English | Meaning |
|------|------|------|
| Residual-stream width | hidden size / dim | token-vector length (Qwen2.5-0.5B is 896) |
| MLP intermediate width | intermediate size | feed-forward network's middle layer (4864) |
| Number of layers | num layers | Transformer repeat count (24) |
| Head | head | attention's parallel unit |
| Head dimension | head dim | each head's vector length (64) |
| Vocabulary size | vocab size | how many tokens it knows (151936) |
| Sequence length | seq len | maximum token count (2048) |
| Token embedding | embedding | the token → vector lookup table |
| Projection | projection | matrix multiply `y = Wx + b` |
| Bias | bias | the constant vector added after a projection |
| Residual connection | residual | `x = x + layer(x)`, lets information skip certain layers |
| Feed-forward network | FFN / MLP | the "independent thinking" submodule inside each layer |
| Decoder | decoder | the Transformer part that generates words left to right |
| Autoregressive | autoregressive | output becomes input again, looping to generate |
| Parameter count | parameters | total elements across all weight matrices (0.5B ≈ 494 million) |
| Tensor | tensor | multi-dimensional array |
| Scalar | scalar | 0-dimensional tensor (a single number) |
| Vector | vector | 1-dimensional tensor (a row of numbers) |
| Matrix | matrix | 2-dimensional tensor (a grid of rows and columns) |
| Feature dimension | feature dim | column count of each token vector (896), independent of the row count (seq_len) |

---

## A.2 Attention Terms

| Term | English | Meaning |
|------|------|------|
| Query | Q | the "what am I looking for" vector of the current token |
| Key | K | the "what am I" vector of each token |
| Value | V | the content vector of each token |
| Attention | attention | lets each word look at other words and decide whom to focus on |
| Attention score | attention score | the relatedness computed by Q·Kᵀ |
| Scaling factor | scale (√d) | prevents the dot-product value from being too large; divide by √head_dim |
| softmax | — | normalizes scores into probabilities (summing to 1) |
| Causal attention | causal attention | only looks left, not right (`for t<=pos`) |
| GQA | Grouped Query Attention | multiple Q heads share KV (Qwen has 14 Q heads sharing 2 KV heads) |
| KV Cache | Key-Value Cache | caches historical K/V to avoid recomputation, drops generation complexity from O(N²) to O(N) |
| RoPE | Rotary Position Embedding | rotary position encoding, encodes position via rotation |
| rope_theta | RoPE base frequency | base frequency; the larger, the farther positions can be encoded (Qwen uses 1e6) |
| Dot product | dot product | multiply corresponding positions of two vectors and sum; measures direction agreement |
| Causality | causality | the current token can only see tokens before it |

---

## A.3 Normalization & Activation Terms

| Term | English | Meaning |
|------|------|------|
| Normalization | normalization | pulls the vector back to a standard magnitude, preventing multi-layer accumulation from exploding |
| RMSNorm | Root Mean Square Norm | divides by the root mean square (one fewer step than LayerNorm — no mean subtraction) |
| LayerNorm | Layer Normalization | subtract mean → divide by standard deviation → multiply by weight (three steps) |
| eps | epsilon | a small constant that prevents division by zero (1e-6) |
| Root mean square | RMS | Root Mean Square, `√(Σx²/n)` |
| SwiGLU | Swish-Gated Linear Unit | gated feed-forward network, `silu(gate) × up` |
| silu | Sigmoid Linear Unit | `x · sigmoid(x)`, smoother than ReLU |
| gate | gating | dynamically decides which channels activate (soft gate, 0 to 1) |
| ReLU | Rectified Linear Unit | `max(0, x)`, a hard gate (either 0 or the original value) |
| Activation function | activation function | a function that introduces non-linearity |
| Activation ratio | active ratio | the fraction of channels above a threshold; reflects MLP sparsity |
| Norm | L2 norm | the "length" of a vector `√(Σx²)`; used in logs to summarize vector magnitude |

---

## A.4 Tokenization Terms

| Term | English | Meaning |
|------|------|------|
| Tokenizer | tokenizer | the bridge between text and token ids |
| token | — | the smallest unit the model processes (a word / a character / a chunk of bytes) |
| token id | — | the integer index of a token in the vocabulary |
| BPE | Byte-Pair Encoding | byte-pair encoding tokenization; repeatedly merges the most frequent adjacent pair |
| pre-token | — | the text fragments split out before BPE (e.g. splitting words by whitespace) |
| byte-level | — | byte → unicode mapping (GPT-2's solution) |
| merge | — | a BPE merge rule |
| vocab | vocabulary | the vocabulary, the set of all tokens |
| BOS | Beginning Of Sequence | sequence-start token |
| EOS | End Of Sequence | sequence-end token (generation stops when this is hit) |
| PAD | Padding | padding token (for batch alignment) |
| Hash table | hash table | uses a hash function to speed up "string → id" lookup |
| FNV-1a | — | a simple fast string hash algorithm |
| Chaining | chaining | a solution that strings collided entries together with a linked list |
| Hash collision | hash collision | different strings hashing to the same value |
| teacher forcing | teacher forcing | in the prefill phase, follow the gold answer instead of sampling |

---

## A.5 Data-Format Terms

| Term | English | Meaning |
|------|------|------|
| safetensors | — | HuggingFace's binary tensor format (safe + mmap friendly) |
| dtype | data type | tensor data type |
| fp32 | float32 | 32-bit standard float (sign 1 + exponent 8 + mantissa 23) |
| bf16 | bfloat16 | 16-bit float (sign 1 + exponent 8 + mantissa 7); fp32 with the low 16 bits chopped |
| fp16 | float16 | 16-bit float (sign 1 + exponent 5 + mantissa 10); smaller dynamic range |
| mmap | memory map | memory-mapped file; accessing memory is equivalent to reading the file |
| little-endian | little-endian | the low byte is stored at the low address |
| header | — | the JSON metadata table at the start of a safetensors file |
| raw buffer | — | the contiguous binary tensor data after the safetensors header |
| data_offsets | — | the `[start, end)` range of a tensor inside the raw buffer |
| Quantization | quantization | approximate weights with fewer bits (e.g. int4) |
| int4 / int8 | — | 4-bit / 8-bit integer representation |
| scale | scale factor | the divisor that compresses the floating-point range into an integer interval during quantization |

---

## A.6 C / Engineering Terms

| Term | Meaning |
|------|------|
| row-major | matrix stored row by row (C default); `W[i*n+j]` is row i, column j |
| strict aliasing | strict aliasing rule; `float*` and `uint32_t*` pointing at the same memory is undefined behavior |
| memcpy | memory copy; the compiler optimizes it to zero overhead (the standard way to bypass strict aliasing) |
| recursive descent | the parser recurses top-down (e.g. the JSON parser calling parse_object when it sees `{`) |
| RFC | internet standard document (e.g. RFC 8259 = the JSON standard) |
| unescape | unescape, restoring `\"` back to `"` |
| surrogate pair | surrogate pair; UTF-16 uses two 16-bit codes to represent a rare character |
| BMP | Basic Multilingual Plane; the Unicode basic plane (code points 0-65535) |
| chaining | string collided hash entries together with a linked list |
| FNV-1a | a string hash algorithm |
| xorshift | a pseudo-random-number algorithm (uses XOR and shifts) |
| PRNG | Pseudo Random Number Generator |
| seed | seed, the PRNG's initial state |
| argmax | take the index of the maximum value |
| roulette wheel | roulette wheel, sampling by probability distribution |
| top-k | keep only the k highest-probability entries |
| temperature | temperature, the parameter that scales logits before softmax |
| softmax | the function that normalizes scores into probabilities |
| ANSI escape code | terminal color-control sequences (`\033[31m` etc.) |
| TTY | TeleTYpewriter, terminal |
| isatty | checks whether a file descriptor is a terminal |
| stderr | standard error stream (logs go here) |
| stdout | standard output stream (generated text goes here) |
| va_list | C variadic-argument mechanism (makes `printf("a=%d", a)` work) |
| va_start / va_end | initialize / finalize va_list iteration |
| vsnprintf | the va_list version of printf |
| ASan | AddressSanitizer, the compiler's built-in memory-error detector |
| UBSan | UndefinedBehaviorSanitizer, undefined-behavior detector |
| sanitizer | generic term for memory / behavior detection tools |

---

## A.7 Ecosystem & Performance Terms

| Term | English | Meaning |
|------|------|------|
| Inference | inference | computing a result with a trained model (forward only, no weight updates) |
| Forward pass | forward pass | compute from layer 1 through layer N |
| Backward pass | backward pass | during training, compute gradients back from the output; this project doesn't do it |
| prefill | prefill | the phase that fills the KV Cache for the entire prompt |
| decode | decode | the phase that autoregressively generates new tokens one slot at a time |
| TTFT | time-to-first-token | the latency to generate the first token (the prefill cost) |
| Continuous Batching | continuous batching | every step batches all active requests together; the GPU never idles |
| PagedAttention | paged attention | slices the KV Cache into fixed-size pages allocated on demand (vLLM innovation) |
| Prefix Caching | prefix caching | caches the KV Cache of shared prefixes to reuse compute |
| RadixAttention | radix-tree attention | SGLang caches prefixes with a radix tree, caching more combinations than a hash table |
| FlashAttention | flash attention | tiled attention computation; VRAM goes from O(seq²) down to O(seq) |
| Speculative Decoding | speculative decoding | small model guesses + large model verifies; one forward pass produces multiple tokens |
| Tensor Parallelism | tensor parallelism | slice a large matrix by rows / columns across multiple cards |
| Pipeline Parallelism | pipeline parallelism | slice by layer across multiple cards; each card handles a few layers |
| CPU Offloading | CPU offloading | weights stay in main memory; move each layer to VRAM when needed |
| MoE | Mixture of Experts | mixture of experts; only a fraction of expert weights are used per token |
| SIMD | Single Instruction Multiple Data | single-instruction multiple-data; CPU parallelism (AVX / NEON) |
| Tensor Core | — | the hardware unit on a GPU purpose-built for matrix multiplies |
| cuBLAS | — | NVIDIA's linear algebra library (the GPU version of BLAS) |
| cuDNN | — | NVIDIA's deep-learning operator library |
| GGUF | — | llama.cpp's model format (includes quantized weights) |
| ggml | — | llama.cpp's self-developed tensor compute library |
| GQA | Grouped Query Attention | multiple Q heads share KV (see A.2) |
| MLA | Multi-head Latent Attention | DeepSeek's attention structure, replacing GQA |
| All-Gather | — | multi-card communication primitive; each card gets all cards' partial results stitched together |
| logits | — | the raw scores projected to the vocabulary at the end (before softmax) |
| Reference implementation | reference implementation | the comparison baseline (this project uses PyTorch transformers) |
| top-5 logits | — | the 5 highest-scoring tokens, used to verify the forward pass |
| tie_word_embeddings | — | the output head reuses the embedding weights (Qwen sets this to true; no lm_head) |

---

## Quick Reference: Key Numbers of This Project

| Item | Value |
|---|---|
| Parameter count | 494,032,768 (≈0.49B, called "0.5B") |
| safetensors file size | 988 MB (bf16 storage) |
| fp32 memory footprint | 1.97 GB |
| dim (feature dim) | 896 |
| hidden_dim (MLP intermediate width) | 4864 |
| n_layers | 24 |
| n_heads (Q heads) | 14 |
| n_kv_heads (KV heads) | 2 |
| head_dim | 64 |
| vocab_size | 151936 |
| seq_len | 2048 |
| rope_theta | 1,000,000 |
| rms_eps | 1e-6 |
| BOS / EOS id | 151643 (`<|endoftext|>`) |
| Speed (CPU fp32) | ~3 tok/s |
| Speed (vLLM A100 concurrent) | ~8000 tok/s |

---

> This glossary is the index for the whole book (Chapters 1-10). Whenever you hit a word you don't understand in any chapter, look it up here first;
> if it's not here, go to the relevant chapter's body text for the detailed explanation.
