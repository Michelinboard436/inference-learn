# Chapter 10 · Ecosystem Comparison & Performance Optimization

> 🌐 [中文版](../10-ecosystem) | English

> Chapter 10 | [Previous: Chapter 9 · Debugging & Verification Methodology](./09-debugging.md) | [Next: Appendix A · Glossary](./appendix-a-glossary.md)

> This is the "closing chapter" of the whole book. Over the previous 9 chapters we built our own engine brick by brick;
> this chapter places it within the broader inference-engine ecosystem — alongside llama.cpp / vLLM / SGLang —
> to see clearly **where we stand, where industrial engines excel, and where the gaps come from**.
>
> This chapter consolidates the project's `COMPARISON.md` (965 lines of ecosystem comparison) and `BOOK.md` Chapter 10.

---

## Table of Contents

- [10.1 One Diagram to Understand the Four Implementations](#101-one-diagram-to-understand-the-four-implementations)
- [10.2 Tech-Stack Layer-by-Layer Comparison](#102-tech-stack-layer-by-layer-comparison)
- [10.3 Why "One Engine to Run All Models" Is Impossible](#103-why-one-engine-to-run-all-models-is-impossible)
- [10.4 Computing Fast vs Scheduling Well](#104-computing-fast-vs-scheduling-well)
- [10.5 Why GPU Instead of CPU](#105-why-gpu-instead-of-cpu)
- [10.6 Where the Performance Gap Comes From](#106-where-the-performance-gap-comes-from)
- [10.7 When the Model Doesn't Fit: 5 Loading Strategies](#107-when-the-model-doesnt-fit-5-loading-strategies)
- [10.8 Reading vLLM's Dependency List](#108-reading-vllms-dependency-list)
- [10.9 Applicable Scenarios per Engine](#109-applicable-scenarios-per-engine)
- [10.10 The Evolution Path from Hand-Written to Industrial Engine](#1010-the-evolution-path-from-hand-written-to-industrial-engine)
- [10.11 What's Specifically Missing: This Project vs vLLM vs SGLang](#1011-whats-specifically-missing-this-project-vs-vllm-vs-sglang)
- [10.12 Core Insights](#1012-core-insights)
- [10.13 One-Sentence Summary](#1013-one-sentence-summary)

---

## 10.1 One Diagram to Understand the Four Implementations

First the big picture. The exact same `Attention(Q,K,V) = softmax(QKᵀ/√d)V` ends up looking wildly different across four implementations:

```
Your engine (hand-written C)   llama.cpp (hand-written C++)   vLLM (PyTorch+CUDA)
      │                     │                          │
      │  For learning       │  Local/edge deployment     │  Datacenter production
      │  ~2000 lines        │  ~50k lines                │  ~200k lines + 30 deps
      │  Pure CPU           │  CPU/GPU quantized         │  NVIDIA GPU mandatory
      │  3 tok/s           │  30 tok/s                │  3000+ tok/s
      │  1 request          │  1 request                 │  100+ concurrent requests
      │                     │                          │
      └───── All computing the same formula: Attention(Q,K,V) = softmax(QKᵀ/√d)V ─────┘
```

Multi-dimensional comparison of the four:

| Dimension | This project (hand-written C) | llama.cpp | vLLM |
|------|----------------|-----------|------|
| **Positioning** | Learn the principles | Personal / local deployment | Production-grade serving |
| **Language** | C99 | C++17 | Python + C++/CUDA |
| **Low-level operators** | Hand-written triple loop | Hand-written + SIMD / quantization | cuBLAS / cuDNN / FlashInfer |
| **GPU support** | ❌ | Partial (Vulkan / Metal) | ✅ CUDA mandatory |
| **Quantization** | ❌ (fp32) | ✅ (int4 / int8) | ✅ (multiple formats) |
| **Concurrency** | ❌ | ❌ | ✅ (continuous batching) |
| **KV Cache management** | Pre-allocated array | Pre-allocated + quantized | PagedAttention (paging) |
| **Code size** | ~2,000 lines | ~50,000 lines | ~200,000 lines |
| **Dependencies** | 0 (libc only) | Very few | 30+ packages |
| **Speed (0.5B)** | ~3 tok/s | ~30 tok/s | ~3000 tok/s* |
| **Hardware requirement** | Any CPU | CPU or GPU | NVIDIA A100 / H100 |

*vLLM speed is based on multi-request batched inference; a single request won't reach it.

> **Core fact**: all four compute **the same mathematical formula**; the gap comes entirely from engineering implementation.
> Once you understand your 2000 lines of C, you understand what vLLM does on a GPU — the same thing, only parallelized and with scheduling layered on top.

---

## 10.2 Tech-Stack Layer-by-Layer Comparison

To see "where the difference is", lay the tech stacks of these three implementations out layer by layer.

### 10.2.1 This Project (Your Hand-Written Engine)

```
┌──────────────────────────────────┐
│  run.c  (CLI entry, generation loop)     │
│  net.c  (forward pass: rmsnorm/rope)     │  ← all hand-written, one layer
│  tokenizer.c (BPE tokenizer)             │
│  safetensors.c (weight loading)          │
├──────────────────────────────────┤
│  libc (malloc/printf/expf)        │  ← the only dependency
├──────────────────────────────────┤
│  CPU hardware                     │
└──────────────────────────────────┘
```

**Characteristics**: all logic lives at **the same abstraction level**. Open `net.c` and from `forward()` you can read the entire computation flow in one glance. No indirect calls, no dynamic dispatch, no CUDA, no PyTorch, no HTTP server.

### 10.2.2 vLLM's Tech Stack

```
┌─────────────────────────────────────────────────────┐
│  User request (OpenAI API format)                          │  ← HTTP interface
├─────────────────────────────────────────────────────┤
│  FastAPI + Starlette                               │  ← web service layer
│  (your engine has no such layer, pure CLI)                  │
├─────────────────────────────────────────────────────┤
│  vLLM scheduler                                    │  ← vLLM core
│    PagedAttention (paged KV cache management)              │
│    Continuous Batching (dynamic batching)                   │
├─────────────────────────────────────────────────────┤
│  transformers (model definition / weight loading)           │  ← model abstraction layer
│  tokenizers (BPE tokenization)                              │  (you hand-wrote the equivalent)
├─────────────────────────────────────────────────────┤
│  PyTorch (torch.matmul, tensor-ops frontend)        │  ← tensor compute layer
├─────────────────────────────────────────────────────┤
│  cuBLAS / cuDNN / FlashInfer / CUTLASS             │  ← CUDA kernel layer
│  (the real GPU-parallel matrix multiplies)                  │
├─────────────────────────────────────────────────────┤
│  NVIDIA GPU (CUDA cores + Tensor cores)             │  ← hardware layer
└─────────────────────────────────────────────────────┘
```

**Characteristics**: **7 layers of abstraction**. A single `model.generate()` call has to pass through Python → PyTorch → C++ → CUDA → GPU. Each layer applies its own optimizations, but together they make "how is this actually computed" opaque.

### 10.2.3 llama.cpp's Tech Stack (the Middle Ground)

```
┌─────────────────────────────────────────────────────┐
│  CLI / server (HTTP)                                │
├─────────────────────────────────────────────────────┤
│  llama.cpp core logic                               │
│    GGUF format parsing (with quantized weights)             │
│    Transformer forward (quantized operators)                │
├─────────────────────────────────────────────────────┤
│  ggml (tensor compute library, hand-written)        │  ← its own cuBLAS
│    CPU: AVX2/NEON SIMD                                      │
│    GPU: Metal (Mac) / Vulkan / CUDA                         │
├─────────────────────────────────────────────────────┤
│  CPU or GPU hardware                                │
└─────────────────────────────────────────────────────┘
```

**Characteristics**: like your engine, it's "written from the bottom up", but it adds a **home-grown tensor library ggml** (essentially writing its own mini cuBLAS), plus quantization and SIMD.

### 10.2.4 Three Layers Compared at a Glance

```
This project   llama.cpp        vLLM
─────────────────────────────────────
1 layer        3 layers         7 layers
Pure C         C++ + ggml       Python + PyTorch + CUDA
0 deps         very few deps    30+ deps
Fully transparent   Partially transparent   Mostly black box
```

More layers = more features and higher performance, but also harder to understand. This project deliberately keeps just one layer so you can see straight through it.

---

## 10.3 Why "One Engine to Run All Models" Is Impossible

You may have noticed: this project's `net.c` is **written specifically for Qwen2.5**. Switching to another model means changing the code. This isn't laziness on our part — it's **the core contradiction of the entire AI inference ecosystem**.

### 10.3.1 Every Model Adds Its Own Twist on the Base Architecture

```
                  Norm        Positional encoding   Attention       QKV bias    Activation   Embedding
Llama-2-7B        RMSNorm   RoPE                  GQA             ❌ none     SwiGLU   not tied
Llama-3-8B        RMSNorm   RoPE(θ=500k)          GQA             ❌ none     SwiGLU   not tied
Qwen2.5 (this project) RMSNorm   RoPE(θ=1M)            GQA             ✅ has      SwiGLU   tied ★
Mistral-7B        RMSNorm   RoPE + sliding window GQA + sliding window ❌ none  SwiGLU   not tied
DeepSeek-V2       RMSNorm   RoPE (decoupled)      MLA (new structure)   special    MoE      partially tied
```

Each model changes something:

- **Qwen** added QKV bias (Llama doesn't have it) — that's why our `net.c` passes `bq/bk/bv`
- **Mistral** added sliding-window attention (only attends to recent tokens, saves memory)
- **DeepSeek-V2** replaced GQA with MLA and swapped MLP for MoE (multiple expert networks)
- **Llama-3** bumped RoPE's theta from 10k to 500k (to support longer context)

### 10.3.2 What These Differences Mean for the Engine

Going from Qwen2.5 to supporting other models, here's what code needs to change:

```
Switch to Llama-3:    🟢 easy (change config numbers + drop QKV bias + load lm_head)
Switch to Mistral:    🟡 medium (add sliding-window logic to attention)
Switch to DeepSeek:   🔴 rewrite (replace attention with MLA, MLP with MoE, decouple RoPE)
```

The more "twists" a model adds, the bigger the engine changes. **DeepSeek's MLA and MoE are brand-new structures — you essentially have to rewrite the attention and MLP halves of `net.c`**.

### 10.3.3 Three Solutions That Emerged From This

| Approach | Representative | How it handles new models | Cost |
|------|------|--------------|------|
| **① Specialized engine** | This project, TensorRT-LLM | Change the code per model | Simple but not general |
| **② Config-driven** | llama.cpp (GGUF) | Read config and auto-adapt via `if/else` | Code is full of branches |
| **③ Computation-graph-driven** | PyTorch, vLLM, ONNX | The model carries its own computation steps; the engine just executes | Engine is extremely complex |

Trade-offs of the three:

```
Approach ① (specialized):   Generality ★          Simplicity ★★★★★     Performance ★★★★★
  Our engine: runs only Qwen2.5, but all 2000 lines are understandable

Approach ② (config):        Generality ★★★        Simplicity ★★★        Performance ★★★★
  llama.cpp:   one engine + GGUF format, auto-adapts by reading config
               code full of if (sliding_window) { ... } else { ... }

Approach ③ (graph):         Generality ★★★★★     Simplicity ★          Performance ★★★
  PyTorch:     implements every possible operator, reads model definition files and executes
               PyTorch has tens of millions of lines because it has to support every structure
```

### 10.3.4 This Explains Two Common Questions

**"Why does vLLM depend on PyTorch?"**
Because PyTorch is approach ③ — it implements every possible operator and can run any model. vLLM stands on PyTorch's shoulders and adds only scheduling optimizations.

**"Why is our engine only 2000 lines?"**
Because we're approach ① — we only support one model, Qwen2.5. We don't need `if/else` to adapt to multiple structures, so the code is extremely minimal.

### 10.3.5 This Contradiction Keeps Driving New Formats

Every time a new model architecture appears (e.g. DeepSeek's MLA), every engine has to follow suit:

- llama.cpp has to update its code to support MLA
- vLLM has to add MLA operators
- ONNX has to define the MLA operator specification

This is why inference engines update so frequently — **it's not that the engines want to change, it's that the models keep changing**.

> **In one sentence**: model architectures bloom in a hundred flowers, and engines must either "customize per model" (fast but not general) or "implement every possibility" (general but complex). There's no having it both ways — this is the fundamental contradiction of the inference-engine ecosystem.

---

## 10.4 Computing Fast vs Scheduling Well

Many people assume vLLM is faster than llama.cpp because of "a better algorithm". It isn't. **The same attention formula — vLLM's real advantage lies in scheduling**.

### 10.4.1 Computing Fast: GPU + Specialized Kernels

This is the most intuitive difference. Take the same matrix multiply:

```c
// Your engine (CPU, single-threaded, scalar loop)
for (int i = 0; i < d; i++) {
    for (int j = 0; j < n; j++) {
        val += wrow[j] * x[j];
    }
}
// Result: 896×896 matrix multiply ≈ tens of microseconds
```

```python
# vLLM / PyTorch (GPU, thousands of cores in parallel)
output = torch.matmul(W, x)   # under the hood calls cuBLAS
# Result: same matrix multiply ≈ a few nanoseconds (1000x faster)
```

A GPU has thousands of cores working at once, plus Tensor Cores purpose-built for matrix multiplies. This isn't "a better algorithm" — it's **hardware parallelism crushing the competition**. This part is "computing fast", and you can add it incrementally in C (quantization, SIMD).

### 10.4.2 Scheduling Well: vLLM's Real Innovation

This is where vLLM earns its money. Consider this scenario:

```
User A: "Write a poem about autumn"     → generates 50 tokens, trickling out
User B: "1+1="                          → generates 1 token, instant reply
User C: "Translate this passage..."     → generates 200 tokens
```

**The traditional approach (your engine / llama.cpp)**:

```
Process A ████████████████████████████ (50 steps)
                                       Process B ██ (1 step)
                                                  Process C ████████████████ (200 steps)
GPU spends most of its time waiting for one request to finish
```

**vLLM's continuous batching**:

```
A ████████████████████████████
B ██                            ← B finishes in 1 step, slot immediately goes to D
C                ████████████████████████████████████████████████
D                   ████████████████████   ← after B finishes, a new request is dynamically inserted
   ↑ every step batches all active requests together, GPU never idles
```

**Effect**: GPU utilization jumps from ~30% to ~95%. Under concurrency, vLLM's throughput is 10-20x that of llama.cpp.

### 10.4.3 PagedAttention: Memory Management for KV Cache

Traditional KV Cache (what your engine uses):

```c
// Pre-allocate a fixed-size array
float *key_cache = calloc(n_layers * seq_len * kv_dim, sizeof(float));
// Problem: no matter how short the sequence, memory is reserved for the max length
// With multiple requests, each hogs its own block — severe waste
```

vLLM's PagedAttention (inspired by the OS's virtual-memory paging):

```
Slice the KV cache into fixed-size "pages" (blocks), allocate on demand:

Traditional (your engine):
Request A [████████░░░░░░░░] 16 slots reserved, only 8 used
Request B [██░░░░░░░░░░░░░░] 16 slots reserved, only 2 used
Waste: 14/32 = 44%

PagedAttention (vLLM):
Page1[AABB] Page2[AAAA] Page3[AACC] ...
Allocated on demand, no waste, memory utilization ~95%
```

This lets vLLM pack more concurrent requests into the same amount of VRAM.

### 10.4.4 The Relationship Among the Three Things

```
"Computing fast" (hardware + operator layer):
  - GPU parallelism
  - SIMD / quantization
  - FlashAttention
  → you can add these incrementally; your engine can also become "fast"

"Scheduling well" (engine architecture layer):
  - Continuous Batching
  - PagedAttention
  - Prefix Caching
  → requires re-architecting the whole inference loop; this is vLLM's moat
```

---

## 10.5 Why GPU Instead of CPU

A fundamental question: our engine runs on CPU (~3 tok/s). Why does the industry use GPUs?
The answer is hidden right inside the Transformer's parameter list — **matrix multiplication is a sea of independent operations, born for parallelism**.

### 10.5.1 How Many Multiplies Does a Single Token Take

Take this project's 0.5B model — for every token generated:

```
Inside each layer:
  Q projection (896×896):    802,816 multiply-adds
  K projection (128×896):    114,688
  V projection (128×896):    114,688
  O projection (896×896):    802,816
  MLP gate (4864×896):       4,358,144   ← big chunk
  MLP up (4864×896):         4,358,144   ← big chunk
  MLP down (896×4864):       4,358,144   ← big chunk
  Per-layer total:           14,909,440

× 24 layers:                 357,826,560
+ logits (896×151936):       136,134,656
────────────────────────────────────
Total:                       493,961,216 ≈ 500 million multiply-adds / per token
```

Scaled up to real large models:

```
Model               Per-token compute      8-core CPU time    A100 time
Qwen2.5-0.5B       0.5 TFLOP              1.2 s              0.0008 s
Qwen2.5-7B         7 TFLOP                17.5 s             0.011 s
Qwen2.5-72B        72 TFLOP               180 s              0.115 s
Llama-3.1-405B     405 TFLOP              1012 s             0.649 s

CPU vs GPU gap: 100-1000x
```

### 10.5.2 Why Is the GPU So Much Faster

The matrix multiply `out[i] = Σ W[i][j] × x[j]` has a key property: **each row's computation is fully independent**.

```
MLP gate 4864×896 matrix × 896-dim vector:

  out[0]    = W[0][:]    · x    ← independent of other rows
  out[1]    = W[1][:]    · x    ← independent of other rows
  out[2]    = W[2][:]    · x    ← independent of other rows
  ...
  out[4863] = W[4863][:] · x    ← independent of other rows

All 4864 rows can be computed at once!
```

CPU vs GPU design differences:

```
CPU (8-16 cores):                GPU (thousands-tens of thousands of cores):
  Goal: a few complex tasks        Goal: a sea of simple tasks
  Each core is powerful,           Each core is simple,
  good at branching logic          good at parallel computation

  Compute 4864 rows: 8 cores in 608 rounds    Compute 4864 rows: 6912 cores in one round
  → slow (queuing)                              → fast (all at once)
```

**A moving-cargo analogy**:

```
4864 boxes to move:
  CPU (8 people):  each moves 608 boxes, take turns in line → slow
  GPU (6912 people): one box each, all done at once         → 600x+ faster
```

### 10.5.3 That's Why Transformers Are Naturally Suited to GPUs

Every Transformer layer is a few large matrix multiplies, and each matrix multiply is thousands of independent row-computations — a perfect match for the GPU's "thousands of cores in parallel".

```
CPU is good at: if/else logic, serial algorithms, operating systems
GPU is good at: matrix multiplies, image processing, neural networks  ← Transformers are all these
```

Our C engine runs on CPU, so the `for (i...) for (j...)` in `net.c` is serial — computing row by row. A GPU would distribute these thousands of rows across thousands of cores and compute them simultaneously. That's the source of the 100-1000x gap.

> This project uses CPU for **learning the principles** (simple, readable code).
> Industrial engines use GPUs for **speed** (thousands-fold parallelism). Once you understand this simplest CPU version, you can understand that the GPU version does the same thing — just parallelized.

---

## 10.6 Where the Performance Gap Comes From

For Qwen2.5-0.5B, single-token generation speed comparison:

```
Your engine (M-series CPU, fp32)   :   3 tok/s
  └─ bottleneck: CPU scalar loop, no parallelism

llama.cpp (M-series, Q4 quant)     :  30 tok/s   (10x)
  └─ gain: int4 quantization (4x memory bandwidth) + NEON SIMD (4x parallelism)

llama.cpp (RTX 4090, Q4)          : 200 tok/s   (66x)
  └─ gain: GPU thousands-of-cores parallelism

vLLM (A100, fp16, single request) : 400 tok/s   (133x)
  └─ gain: A100 stronger than 4090 + FlashAttention

vLLM (A100, fp16, 100 concurrent) : 8000 tok/s  (2666x total throughput)
  └─ gain: continuous batching packs 100 requests into one super-batch
```

**Breaking down the 2666x**:

| Optimization | Multiplier | Where it's done |
|---------|------|---------|
| int4 quantization | ×4 | llama.cpp's operator layer |
| SIMD (NEON / AVX) | ×4 | llama.cpp's operator layer |
| GPU parallelism (CPU→GPU) | ×7 | Hardware switch |
| FlashAttention | ×2 | CUDA kernel optimization |
| Continuous Batching (concurrency) | ×12 | vLLM's scheduling layer |
| **Total** | **×2688** | |

Note: **the first 4 optimizations are "computing fast", which your engine can add incrementally** (quantization and SIMD are both doable in C).
**The last one is "scheduling well", which requires re-architecting the whole inference loop** — that's vLLM's moat.

---

## 10.7 When the Model Doesn't Fit: 5 Loading Strategies

Everything above is about small models like 0.5B (2GB of weights, just stuff it into memory).
But real-world large models are routinely tens or hundreds of GB, while **a single GPU's VRAM is only a few tens of GB** — they don't fit.
This is the core challenge of large-model engineering, and the industry has 5 solutions.

### 10.7.1 How Big Is the Problem

```
Model                Parameters   fp16 weights   Fits in one card?
──────────────────────────────────────────────────────
Qwen2.5-0.5B (here)  0.5B         1 GB           ✅ easily
Qwen2.5-7B           7B           14 GB          ✅ one RTX 4090
Qwen2.5-72B          72B          144 GB         ❌ max single card 141GB, barely
Llama-3.1-405B       405B         810 GB         ❌ needs 6 H100s
GPT-4 (est.)         1.8T         3.6 TB         ❌ needs dozens of cards

GPU VRAM reference:
  RTX 4090 (consumer)      24 GB    ¥16k
  A100 80G (datacenter)    80 GB    ¥180k
  H200 (latest large VRAM) 141 GB   ¥300k
```

**Core contradiction**: models keep growing; VRAM can't keep up. 405B needs 825GB, the largest single card has 141GB — a 6x gap.

### 10.7.2 Strategy ①: Quantization Compression (most common)

**In one sentence**: use fewer bits to approximately represent the same weights, trading a little precision for smaller size and faster speed.

#### Same Model at Different Precisions

```
Same 72B model:
  fp32 (32-bit): 288 GB  ← most accurate, rarely used
  fp16 (16-bit): 144 GB  ← training precision
  int8  (8-bit):  72 GB  ← half the size, small accuracy loss
  int4  (4-bit):  36 GB  ← 4x compression, mainstream inference precision ★
  int2  (2-bit):  18 GB  ← 8x compression, large accuracy loss
```

#### Why Quantization Works: Models Pick Words by "Direction", Not Exact Values

As covered earlier, when the model finally predicts a word it uses **cosine similarity (direction closeness)**, not exact numerical values.
So a small error in an individual weight is fine, as long as the overall direction stays the same:

```
Same weight, different precisions:
  fp32: 0.034123001  ← 9 significant digits
  int4: 0.03         ← 2 significant digits

Looks like a big error? But after quantizing 1000 weights, the overall direction barely changes:
  fp16: direction similarity 1.000000  ← no change
  int8: direction similarity 0.999970  ← almost no change
  int4: direction similarity 0.991512  ← still 0.99+
  int2: direction similarity 0.837232  ← this is where the loss becomes obvious

→ int4 has little impact because the model picks words by direction
→ int2 is where quality visibly degrades
```

#### How Quantization Works: Map to Fixed Discrete Values

int4 has only 4 bits, representing 16 values (-8 to +7). Map continuous floating-point weights onto these 16 values:

```
Original fp16 weights (8 of them):
  [-0.42, -0.15, -0.03, 0.08, 0.21, 0.35, 0.41, -0.28]

Step 1: compute the scale factor
  scale = max absolute value / 7 = 0.42 / 7 = 0.06
  (compress the [-0.42, 0.42] range into the [-7, +7] integer interval)

Step 2: divide each weight by scale, round to integer
  -0.42 / 0.06 = -7.0 → -7
  -0.15 / 0.06 = -2.5 → -2
   0.08 / 0.06 = +1.3 → +1
   0.41 / 0.06 = +6.8 → +7
  ...

Step 3: store only these small integers (4 bits each)
  stored: [-7, -2, 0, 1, 4, 6, 7, -5]

At inference: integer × scale recovers an approximation
  -7 × 0.06 = -0.42  (original -0.42, error 0.00)
  -2 × 0.06 = -0.12  (original -0.15, error 0.03)
  ...
  Average error 0.02, direction barely changes
```

#### The Dual Benefit of Quantization

**Benefit ①: saves storage / VRAM** (the most intuitive one)

```
fp16 → int4: each weight shrinks from 16 bits to 4 bits, 4x smaller
  72B model: 144GB → 36GB → fits in a single A100 80G!
  405B model: 810GB → 200GB → 3 A100s can run it
```

**Benefit ②: computes faster** (the more important one, often overlooked)

```
The bottleneck of matrix multiply is "the speed of reading weights" (memory bandwidth), not the computation itself
  int4 weights are 4x smaller → in the same time you can read 4x more weights → 4x faster
  And GPU int4 throughput is higher than fp16 (Tensor Cores are specially optimized)

Combined: 4-8x faster

Cargo analogy:
  fp16: each box 16 kg, 80 boxes per truck
  int4: each box 4 kg,  320 boxes per truck  ← 4x per truck
```

#### The Cost

Slightly lower accuracy, but experiments show **int4 has little impact on most tasks** (because direction barely changes).
Only extreme quantization (int2) noticeably degrades quality.

#### Does This Project Use Quantization?

**No**. This project uses fp32 (most accurate, 4 bytes per number), for these reasons:

- A learning project needs precise numbers (for comparing against PyTorch)
- 0.5B is only 2GB, no need to quantize
- Quantized operators (int4 decode + compute) are harder to write than fp32

If you wanted to add a quantized version, you'd decode the weights from int4 to fp32 inside `matmul` before computing. llama.cpp's Q4_K_M format does exactly this.

### 10.7.3 Strategy ②: Tensor Parallelism — Slice the Matrix

**Principle**: slice a large matrix **by rows or by columns**, putting each slice on a different card.

```
405B's Wq matrix, sliced by rows across 8 cards:

  The whole Wq (16384 rows):
    ┌─────────────────┐
    │   GPU 0's part  │ ← 2048 rows
    ├─────────────────┤
    │   GPU 1's part  │ ← 2048 rows
    ├─────────────────┤
    │   ...           │
    ├─────────────────┤
    │   GPU 7's part  │ ← 2048 rows
    └─────────────────┘

  Input x is broadcast to all 8 cards
  Each card computes only its slice → produces a partial result → All-Gather stitches them
```

**Effect**: each card stores only 1/N of the weights. 825GB / 8 ≈ 103GB/card (fits on A100).

**Cost**: after each layer the cards have to communicate to stitch results (All-Gather), adding latency. More cards = more communication overhead.

> This project is single-threaded and has no such concept. Once you understand matmul's row loop, you understand row-sliced tensor parallelism.

### 10.7.4 Strategy ③: Pipeline Parallelism — Slice by Layer

**Principle**: slice by layer — **each card handles a few layers**.

```
24-layer model, 2 cards:

  GPU 0 (holds weights for 12 layers)   GPU 1 (holds weights for 12 layers)
  ┌───────────────────┐                 ┌───────────────────┐
  │ Layer 0  → 11     │ pass along →    │ Layer 12 → 23     │
  └───────────────────┘                 └───────────────────┘

  Data flow: GPU0 computes layers 0-11 → result passed to GPU1 → GPU1 computes layers 12-23
```

**Effect**: weights split in half, each card stores half.

**Cost**: there are "bubbles" — while GPU0 computes, GPU1 waits, utilization is not full. Can be mitigated with micro-batching.

> In this project's `forward()`, the `for (l = 0; l < L; l++)` loop has each layer independent, so it's naturally sliceable across cards.

### 10.7.5 Strategy ④: CPU Offloading — Fetch on Demand

**Principle**: VRAM can't hold everything, but **server main memory is large** (often 1TB). Keep weights in main memory and move each layer to VRAM only when needed.

```
  Main memory (1TB, cheap)         VRAM (80GB, expensive)
  ┌───────────────────┐            ┌───────────────────┐
  │ Layers 0-23, all weights │ fetch on demand → │ the layer being computed now │
  └───────────────────┘            └───────────────────┘

  Compute Layer 0: move from memory to VRAM → compute → discard
  Compute Layer 1: move from memory to VRAM → compute → discard
  ...
```

**Effect**: VRAM only needs to hold one layer — any large model can run.

**Cost**: moving is slow (PCIe bandwidth is far below VRAM bandwidth), inference slows down 10-100x.

**Use case**: when you can't afford multiple cards but still want to run a large model. This project uses `mmap` to map weights into memory — essentially the CPU version of the same idea.

### 10.7.6 Strategy ⑤: MoE Expert Parallelism — Use Only a Fraction

**Principle**: DeepSeek-V2, Mixtral, GPT-4 all use MoE (Mixture of Experts) — **each token uses only a fraction of the weights**.

```
Traditional model: every word passes through all parameters → all must fit in VRAM
MoE model: has 256 "experts", for each word the router picks only 8

  ┌─────────────────────────┐
  │ Router: which expert does this word go to? │
  └────────────┬────────────┘
               ↓
  Expert1  Expert2  Expert3 ... Expert256
  (used)   (used)   (not)        (not)

  A single word uses only 8/256 = 3% of the parameters!
```

**Effect**: total parameters 534B but only 37B active per token — fast.

**Cost**: the total parameters still all need to be stored (but you can swap them in and out + quantize).

### 10.7.7 How They Combine in Practice

Deploying a large model usually **stacks multiple methods**:

```
Deploying a 405B model (825GB fp16), a typical plan:

  ① Quantize to int4:    825GB → 206GB   (4x compression)
  ② Tensor parallel 8 cards: 206GB → 26GB/card (8 slices)
  ③ One 8×A100 server: each card 26GB < 80GB ✓

  Result: one 8-card server (¥1.5M) can run 405B
```

**The 5 strategies compared**:

| Strategy | Principle | Compression | Cost |
|------|------|--------|------|
| **Quantization** | Fewer bits per weight | ×4-8 | Slight accuracy loss |
| **Tensor parallelism** | Slice matrix across cards | ×N cards | Inter-card communication |
| **Pipeline parallelism** | Slice layers across cards | ×N cards | Has bubbles |
| **CPU offloading** | Weights in memory, fetch on demand | arbitrary | 10-100x slower |
| **MoE** | Use only some experts per token | ×10-30 | Total weights still need storing |

These methods **are not mutually exclusive — they combine**. vLLM is popular because it **automates this combination** (auto-selects quantization, auto-shards, auto-schedules).

> **What this project uses**: full fp32, mmap'd into memory all at once, no sharding. Because 0.5B is only 2GB, any modern computer handles it easily.
> Small models use the simplest approach; large models need these complex techniques. Understanding the simplest case gives you the foundation to look at parallelization techniques.

---

## 10.8 Reading vLLM's Dependency List

vLLM's `requires_dist` on PyPI (the real dependency list), grouped by function:

### Compute Core

| Dependency | Purpose | Your engine's counterpart |
|------|------|---------------|
| `torch==2.11.0` | Tensor-ops frontend | `net.c` hand-written loops |
| `flashinfer-python` | FlashAttention and other attention kernels | `net.c`'s attention loop |
| `nvidia-cudnn-frontend` | NVIDIA deep-learning operators | `net.c`'s rmsnorm/silu |
| `nvidia-cutlass-dsl` | Matrix-multiply template library | `net.c`'s matmul |
| `quack-kernels` | Quantization kernels | (you don't quantize) |
| `humming-kernels` | Inference-specific kernels | - |

### Models / Tokenization

| Dependency | Purpose | Your engine's counterpart |
|------|------|---------------|
| `transformers>=5.5.3` | Model definition / weight loading | `safetensors.c` |
| `tokenizers>=0.21.1` | BPE tokenization | `tokenizer.c` |
| `safetensors>=0.6.2` | Weight file parsing | `safetensors.c` (hand-written) |
| `sentencepiece` | Another tokenizer | - |
| `tiktoken` | OpenAI's tokenizer | - |

### Serving / API

| Dependency | Purpose | Your engine's counterpart |
|------|------|---------------|
| `fastapi[standard]` | HTTP API framework | ❌ (you use CLI) |
| `starlette` | ASGI framework | - |
| `openai>=2.0.0` | OpenAI-compatible client | - |
| `prometheus_client` | Metrics monitoring | - |

### Structured Output

| Dependency | Purpose |
|------|------|
| `xgrammar` | Constrain generation with a grammar (JSON mode) |
| `outlines_core` | Same idea, a different approach |
| `llguidance` | Same |
| `lm-format-enforcer` | Same |

> Just for "structured output" there are 4 libraries to choose from — that's the complexity of production-grade needs.

**For comparison**, your engine's dependency tree is:

```
libc (malloc, printf, expf, mmap, open)
```

That's it. vLLM has **30+**. This isn't vLLM being bloated — it's that what production needs (API serving, concurrency, monitoring, quantization, multiple model formats…) is far more complex than learning the principles.

---

## 10.9 Applicable Scenarios per Engine

| Need | Recommendation | Why |
|------|------|--------|
| **Learning LLM principles** | ✅ This project | 2000 lines, all visible, numbers verified |
| **Local chat (Mac)** | llama.cpp | Has Metal acceleration + quantization, fast |
| **Local chat (with GPU)** | llama.cpp / Ollama | Same; Ollama is a friendlier wrapper |
| **Single-machine API service** | TGI / llama.cpp server | Lightweight HTTP interface |
| **Datacenter high concurrency** | vLLM | PagedAttention + batching crush it |
| **Embedded / edge** | llama.cpp / a slimmed version of this project | Resource-constrained environments |
| **Research experiments** | transformers (PyTorch) | Flexible, easy to modify model structure |
| **Extreme performance** | TensorRT-LLM | NVIDIA official, faster than vLLM but harder to use |

---

## 10.10 The Evolution Path from Hand-Written to Industrial Engine

If you wanted to take the current hand-written engine and gradually turn it into a "usable" engine, here's the path:

### Tier 1: Performance Optimization (keep the architecture)

```
Where you are ──→ add int4 quant ──→ add NEON SIMD ──→ add multithreading
  3 tok/s           12 tok/s           48 tok/s          ~150 tok/s (8 cores)
```

All of this is doable in C; `net.c`'s structure stays basically the same.

### Tier 2: Feature Expansion

```
Add HTTP API (libcurl or hand-written) ──→ support streaming output ──→ support chat templates
Add GGUF format support ──→ support more models (Llama/Mistral)
Add GPU support (Metal/CUDA) ──→ another 10x speedup
```

At this point your engine is roughly the seed of a llama.cpp.

### Tier 3: Production-Grade Scheduling (the hardest)

```
Add paged KV cache management ──→ continuous batching ──→ you've reimplemented vLLM
```

This step requires re-architecting the entire inference loop — from "one request at a time" to "a batch of requests at a time", and the attention computation has to grow a batch dimension. This is months of engineering work.

---

## 10.11 What's Specifically Missing: This Project vs vLLM vs SGLang

The previous sections covered principle-level differences. This one gives a **concrete feature list and performance numbers**, so you know exactly "which parts you're still missing for a production engine".

### 10.11.1 Feature-Gap List

```
Category        Feature                This project  vLLM        SGLang
─────────────────────────────────────────────────────────────────
Core inference  Forward pass           ✅ CPU        ✅ GPU      ✅ GPU
                BPE tokenizer          ✅            ✅          ✅
                Sampling (temp/topk)   ✅            ✅+topp     ✅+topp

Concurrency/    Single request         ✅            ✅          ✅
scheduling      Multi-request          ❌            ✅          ✅
                Continuous Batching    ❌            ✅ ★        ✅ ★
                KV Cache paging        ❌            ✅ PagedAttn ✅ PagedAttn
                Prefix Caching         ❌            ✅          ✅ RadixAttn ★

Performance     GPU acceleration       ❌            ✅ CUDA     ✅ CUDA
optimization    Quantization (int4/int8) ❌          ✅ many    ✅ many
                Flash Attention        ❌            ✅          ✅
                Tensor parallel (multi-card) ❌      ✅          ✅
                Speculative Decoding   ❌            ✅          ✅

Serving/API     HTTP API               ❌            ✅ FastAPI  ✅ FastAPI
                OpenAI-compatible      ❌            ✅          ✅
                Streaming output (SSE) ❌            ✅          ✅
                Multimodal (img/audio/video) ❌      ✅          ✅

Advanced        Structured output (JSON) ❌           ✅          ✅ ★strong
generation      Function Calling       ❌            ✅          ✅
                LoRA adapters          ❌            ✅          ✅

Model support   Architecture variety   1 (Qwen)      dozens      dozens
                Quantization formats   none          10+         10+
```

### 10.11.2 Three Categories of Gap, Explained

#### Category 1: Scheduling & Concurrency (the core gap)

This is our most fundamental gap vs vLLM/SGLang — **we can only handle one request at a time**.

**Continuous Batching**:

```
Our engine (serial):
  Request A ████████████████ (50 tokens)
                            Request B ██ (1 token)
                                       Request C ██████████
  → three requests queue up; compute power is idle most of the time

vLLM/SGLang (continuous batching):
  Request A ████████████████
  Request B ██                   ← B finishes, slot immediately goes to a new request
  Request C      ████████████
  Request D         ███████████  ← dynamically inserted
  → every step batches all active requests together
  → compute is always saturated, utilization 30% → 95%
```

In essence: our `forward()` processes one token sequence at a time. vLLM rewrites it to process multiple sequences at once (adding a batch dimension), so compute is shared across requests.

**PagedAttention (paged KV Cache)**:

```
Our KV Cache (fixed allocation):
  each request reserves [2048 positions × 128 dim] of space
  → even if only 50 tokens are generated, it occupies 2048's worth of memory — 97% waste

vLLM/SGLang (PagedAttention):
  KV Cache sliced into fixed-size "pages" (blocks), allocated on demand
  → generating 50 tokens only occupies 50 positions of memory
  → same VRAM fits more concurrent requests
```

**Prefix Caching**:

```
Two users both ask "Explain quantum mechanics" (same prefix):
  User 1: "Explain quantum mechanics... what is the Schrödinger equation"
  User 2: "Explain quantum mechanics... what are its applications"

Our engine: both requests recompute the prefix's KV Cache from scratch
SGLang (RadixAttention): after the first request finishes, the prefix KV Cache is cached
                          the second request reuses it directly → several times faster
```

#### Category 2: Performance Optimization Techniques

**GPU + CUDA** (the biggest source of performance):

```
Our matmul (CPU serial):  for(i) for(j) one at a time
vLLM's matmul (GPU parallel): torch.matmul → cuBLAS → 6912 cores at once
Gap: 100-1000x (see Section 10.5 for details)
```

**Flash Attention**:

```
Standard Attention: the full [seq_len, seq_len] matrix lives in VRAM
  at seq_len=4096 a single matrix is 64MB; across layers and heads → VRAM explodes

Flash Attention: tiled computation, never stores the full matrix
  → VRAM goes from O(seq_len²) down to O(seq_len)
  → 2-3x faster
```

**Speculative Decoding**:

```
Standard generation: every token runs the full 24-layer forward
Speculative decoding:
  1. a small model quickly guesses 4 tokens
  2. the large model verifies all 4 in one pass (cost ≈ computing 1)
  3. the correct guesses are used directly
  → one forward pass produces 2-3 tokens, 2-3x faster
```

**Quantization**: see Section 10.7.2 for details. int4 makes weights 4x smaller and 4-8x faster.

#### Category 3: Serving & Ecosystem

This part is about "can it be used", not "is it fast":

| Feature | Us | vLLM/SGLang | Purpose |
|------|------|-------------|------|
| HTTP API | ❌ | ✅ | Others can call it over the network |
| OpenAI-compatible | ❌ | ✅ | Drop-in replacement for OpenAI clients |
| Streaming output | ❌ | ✅ | Return as you generate (typing effect) |
| Structured output | ❌ | ✅ | Force valid JSON output |
| Multimodal | ❌ | ✅ | Support image / audio input |
| LoRA | ❌ | ✅ | Load fine-tuned weights |

### 10.11.3 How Big Is the Performance Gap

Same Qwen2.5-7B:

```
Engine          Hardware        Single-request speed   Concurrent throughput
─────────────────────────────────────────────────────────
This project (CPU)   8-core CPU      ~3 tok/s          3 tok/s
llama.cpp            RTX 4090        ~80 tok/s         80 tok/s
vLLM                 A100 × 1        ~150 tok/s        ~3000 tok/s ★
SGLang               A100 × 1        ~150 tok/s        ~4000 tok/s ★
```

**Key gaps**:

```
Single-request speed:   this project 50x slower (CPU vs GPU)
Concurrent throughput:  this project 1000x slower (serial vs batching) ← biggest gap
```

### 10.11.4 vLLM vs SGLang (comparing two top-tier engines)

Both have PagedAttention, Continuous Batching, quantization, multi-card, HTTP API. The differences:

**Where SGLang is stronger**:

- **RadixAttention**: caches prefixes with a radix tree, can cache more combinations than vLLM's hash table → 2-5x faster in multi-turn dialogue and few-shot scenarios
- **Structured output**: native integration + compile-time optimization → 10x+ faster when forcing JSON output
- **Programmatic generation (DSL)**: unique to SGLang, lets you describe the generation flow in Python code (branching generation, picking the best)
- **More aggressive scheduling**: more aggressive kernel fusion → 10-30% higher overall throughput

**Where vLLM is stronger**:

- **More mature ecosystem**: larger community, more docs, more plugins
- **Broader hardware support**: NVIDIA / AMD / Intel / TPU
- **Stability**: proven at large-scale production

### 10.11.5 But These Aren't Defects — They're Different Design Goals

```
This project (learning engine):
  Goal: make you understand every part of an inference engine
  Value: 2000 lines, all understandable, numbers verified
  Benchmark: Karpathy's llama2.c

vLLM / SGLang (production engines):
  Goal: high concurrency, high throughput, low latency
  Value: handle real traffic
  Benchmark: commercial API backends
```

**Our engine deliberately cuts 90% of features** — not out of laziness, but so the remaining 10% (the core algorithms) is clear enough. vLLM has 200k lines and 30+ dependencies; we have 2000 lines and 0 dependencies. The former can go to production; the latter can fit in your head.

---

## 10.12 Long Context: Both Model and Hardware Are Essential

You may have noticed that different models claim wildly different context lengths: GPT-2 is 1024, Qwen2.5 is 32K, Claude 3 is 200K, Gemini 1.5 is 1M. **Supporting long context isn't only about hardware — the model itself must support it too** — both are indispensable.

### 10.12.1 Model Side: Position Encoding Must Reach That Far

This is the most fundamental prerequisite. Recall RoPE from Chapter 5 — position encoding distinguishes positions via rotation angles:

```c
float freq = powf(theta, (float)(-2*i) / head_dim);
float angle = pos * freq;   // rotation angle for position pos
```

**The problem**: if `pos` is large (say 1 million), the rotation angles of different positions get very close together — the difference between adjacent positions becomes so small the model can't tell them apart. It's like measuring millimeters with a ruler whose finest marks are centimeters.

How different models cope:

```
Model              rope_theta    Max context   Position encoding adequate?
─────────────────────────────────────────────────────────
GPT-2              10,000        1,024         ✓ yes
Llama 2            10,000        4,096         ✓ yes
Qwen2.5            1,000,000     32,768        ✓ yes (large theta, reaches far)
Llama 3            500,000       128K          ✓ yes
Claude 3 (YaRN)    ? + scaling   200K          ✓ via RoPE scaling
Gemini 1.5         ? + memory compression  1M  ✓ via memory compression

↑ The larger rope_theta, the smaller the frequency difference between adjacent positions,
  the farther out positions can be encoded. But too large sacrifices short-range
  discrimination — a balance is needed.
```

**If the model's position encoding doesn't support 200K, hardware alone won't help** — the model literally "can't see clearly" the difference between the 150,000th token and the 150,001st.

Additional techniques for long-context models:
- **YaRN / RoPE Scaling**: interpolate over distant positions so the original RoPE can extrapolate further
- **Memory compression** (Gemini): compress early tokens into a summary rather than truly storing a million KV entries
- **Sliding window** (Mistral): precisely retain only the most recent N tokens, fuzz the rest

### 10.12.2 Hardware Side: Can the KV Cache Fit

Assume the model side is fine (position encoding is adequate) — **whether the hardware can hold the KV Cache is the second threshold**.

KV Cache memory = 2 × n_layers × seq_len × n_kv_heads × head_dim × 4 bytes

```
Model             seq_len      KV Cache memory   Intuitive comparison
──────────────────────────────────────────────────────
Qwen2.5-0.5B      2,048        50 MB             ← what this project uses
Qwen2.5-0.5B      32,768       805 MB            ← the model itself is only 2GB!
Qwen2.5-0.5B      1,000,000    24.6 GB           ← even a 0.5B model needs 24GB?!
Qwen2.5-7B        200,000      22.9 GB
Qwen2.5-72B       1,000,000    655 GB            ← 4x larger than the model weights!
Llama-3-405B      1,000,000    1,032 GB          ← the cache alone is 1TB!
```

**KV Cache memory is linear in seq_len** — double seq_len, double memory. This is why long context is so expensive: it's not that the model can't compute, it's that **memory can't hold it**.

### 10.12.3 Compute Cost: Attention Grows Quadratically

The attention score matrix is `[seq_len, seq_len]`, so compute is **O(seq_len²)**:

```
seq_len = 2,048:     4 million scores
seq_len = 32,768:    1 billion scores      ← 256x
seq_len = 200,000:   40 billion scores     ← 10,000x
seq_len = 1,000,000: 1 trillion scores     ← 250 million x!
```

That's why **Flash Attention** is needed — it brings O(n²) memory down to O(n), otherwise just storing the attention matrix blows up VRAM.

### 10.12.4 How Industrial Engines Make 1M Possible

Model support + big enough hardware still isn't enough — the inference engine also has to apply engineering optimizations:

| Technique | What it does | Who uses it |
|------|--------|------|
| **PagedAttention** | Paged KV Cache management, allocate on demand, don't pre-allocate the whole seq_len | vLLM / SGLang |
| **KV Cache quantization** | Quantize the KV Cache too (fp16→int4), 4-8x less memory | vLLM / SGLang |
| **Flash Attention** | Tiled attention computation, memory from O(n²) to O(n) | All GPU engines |
| **Ring Attention** | Distribute ultra-long attention across multiple cards | Gemini / Llama 3 |
| **Sliding window** | Only attend to the most recent N tokens, ignore the rest | Mistral |
| **Memory compression** | Compress early tokens into a summary | Gemini 1.5 |

### 10.12.5 This Project's seq_len

This project uses `seq_len = 2048` (hardcoded in `run.c`). Changing it to 32768 takes one line:

```c
qwen25_0_5b_config(&cfg, 32768);   // change from 2048 to 32768
```

Memory change:

```
seq_len=2048:   KV Cache 50MB,  total memory 2.0GB
seq_len=32768:  KV Cache 805MB, total memory 2.8GB  ← fine, ordinary computers can run it
seq_len=200000: KV Cache 4.9GB, total memory 6.9GB  ← memory gets tight
```

The model itself (Qwen2.5-0.5B) supports 32768 — its RoPE theta=1,000,000 easily covers that distance. Our engine just **doesn't allocate that large a cache** — it's an engineering choice, not a model limitation.

### 10.12.6 Summary

```
Supporting long context = model support + big enough hardware + engine optimization

Model side:
  Large enough RoPE theta / YaRN scaling / memory compression
  → position encoding can distinguish distant tokens

Hardware side:
  Enough VRAM to hold the KV Cache (linear growth)
  + enough compute for attention (quadratic growth)

Engine side:
  PagedAttention / Flash Attention / KV quantization / multi-card distribution
  → turns "theoretically possible" into "actually runnable"
```

**In one sentence**: the gap between 200K and 1M isn't a single factor — the model's position encoding, the hardware's VRAM, the engine's optimizations are all indispensable. Gemini 1M is an engineering miracle that pushes all three to the extreme.

---

## 10.13 Core Insights

### 10.13.1 vLLM's Real Value Isn't the Algorithm, It's the Scheduling

Many people first hearing about vLLM assume it uses some "better attention algorithm". It doesn't — **it computes the same mathematical formula as your hand-written engine**:

```
Attention(Q,K,V) = softmax(QKᵀ/√d)V
```

vLLM's real innovations are at the scheduling layer:

- **PagedAttention** pages the KV Cache, taking memory utilization from 50% to 95%
- **Continuous Batching** packs multiple requests into one batch, taking GPU utilization from 30% to 95%

Neither has anything to do with "computing faster" — same GPU, same operators, just letting **multiple requests share compute**. This is an engineering innovation, not a mathematical one.

### 10.13.2 Your Engine and vLLM Compute Exactly the Same Math

| Question | Answer |
|------|------|
| Do your engine and vLLM compute the same math? | **Exactly the same** (same attention formula) |
| Why is vLLM 1000x faster? | GPU parallelism + quantization + scheduling optimization |
| What's vLLM's most core innovation? | **PagedAttention + Continuous Batching** (scheduling, not algorithm) |
| Does your engine have value? | **Yes** — it's the foundation for understanding all of the above |
| Should I use vLLM or hand-write? | **Depends on the goal**: learning → hand-write, production → vLLM |

### 10.13.3 Why Understanding the Principles Matters

```
Using vLLM without understanding the math:
  → tuning by trial, fixing problems by restarting, speeding up by copying configs
  → vLLM is a black box to you

Using vLLM after understanding the hand-written engine:
  → you know what each parameter changes
  → when things break you can localize to the operator layer
  → you can judge which engine fits which scenario
  → vLLM is a transparent tool to you
```

That's why the first 9 chapters of this book walk you through hand-writing an engine — not to replace vLLM, but to **let you truly understand what vLLM is doing**.

---

## 10.14 One-Sentence Summary

| Question | Answer |
|------|------|
| Do we compute the same math as industrial engines? | **Exactly the same** — same attention formula |
| Where does the gap come from? | "Computing fast" (GPU/quantization/SIMD/FlashAttn, can be added incrementally) + "scheduling well" (Continuous Batching/PagedAttention, vLLM's moat) |
| What if the model doesn't fit? | Quantization, tensor parallelism, pipeline, CPU offloading, MoE — five strategies used in combination |
| What's vLLM's most core innovation? | **Scheduling**, not algorithm |
| Does your engine have meaning? | **Yes** — it's the foundation for understanding all of the above |

**vLLM uses PyTorch, and not only PyTorch — it stands on the shoulders of PyTorch + cuBLAS + FlashInfer + a self-developed scheduler. Your engine is built from the ground up by yourself. For learning purposes, your approach reveals the principles; for production purposes, vLLM's approach is what withstands real traffic.**

> After reading this chapter, you should be able to answer: "if I had infinite time and resources, how would I turn my 2000-line C engine into vLLM?"
> The answer is in Section 10.10's evolution path — three tiers, each clearly decomposable.
> That's the greatest value of writing a hand-written engine once: **you no longer treat vLLM as a black box, but know what it's doing at every step and why**.
