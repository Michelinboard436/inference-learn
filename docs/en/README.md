# 📖 Table of Contents

> A book about understanding LLM inference engines from scratch. ~2700 lines of C, zero dependencies, loads Qwen2.5-0.5B.

<div align="center">

| | |
|:---:|:---:|
| **Code** | ~2,700 lines C99 |
| **Dependencies** | 0 (libc only) |
| **Docs** | 10 chapters + 2 appendices |
| **Model** | Qwen2.5-0.5B (0.49B params) |
| **Accuracy** | Error < 0.0002 vs PyTorch |

</div>

---

## 🗺️ How to Read This Book

Three reading paths — pick what fits:

| 🚀 Quick Start | 📚 Full Course | 🔬 Core Only |
|:-:|:-:|:-:|
| **2-3 hours** | **~1 week** | **1-2 hours** |
| [Ch.1 Basics](./01-basics) → [Ch.2 Weights](./02-weights) → [Ch.4 Model](./04-model) → [Ch.5 Forward](./05-forward) → [Ch.6 Tokenizer](./06-tokenizer) → [Ch.7 Sampling](./07-sampling) | [Start from Ch.1](./01-basics) → read all 10 chapters with source code | [Ch.4 Model](./04-model) → [Ch.5 Forward ★](./05-forward) — the two most important chapters |

---

## Part 1 · Fundamentals

### [Chapter 1 · Basics](./01-basics)
What is an inference engine, tensors and dimensions, how "0.5B" is calculated, what parameters each layer has.

### [Chapter 2 · Weights Storage & Loading](./02-weights)
Safetensors file format, mmap zero-copy loading, bf16→fp32 conversion, embedding table location.

### [Chapter 3 · JSON Parser](./03-json)
Why write your own JSON parser, recursive descent, JsonValue tree structure.

---

## Part 2 · Core

### [Chapter 4 · Model Architecture](./04-model)
Config / TransformerWeights / RunState — the three core structs. Every field in net.h explained.

### [Chapter 5 · Forward Pass ★](./05-forward)
**The most important chapter.** RMSNorm, matmul, RoPE, Attention, GQA, SwiGLU, KV Cache, residual connections — all 6 operators broken down with real numbers.

### [Chapter 6 · Tokenizer BPE](./06-tokenizer)
How text becomes tokens, BPE merge algorithm, byte-level mapping, pre-tokenization.

### [Chapter 7 · Sampling & Generation](./07-sampling)
Prefill + decode two-phase generation, temperature / top-k sampling, roulette wheel, why prefill is compute-bound and decode is memory-bound.

---

## Part 3 · Advanced

### [Chapter 8 · Logging & Visualization](./08-logging)
Tiered logging (-v / -vv), colored output, HTML visualization reports.

### [Chapter 9 · Debugging & Verification](./09-debugging)
Why numerical verification is essential, single-token verification, ASan for memory bugs, layer-by-layer dump.

### [Chapter 10 · Ecosystem Comparison](./10-ecosystem)
How your engine compares to llama.cpp / vLLM / SGLang. GPU vs CPU, quantization, long context.

---

## 📚 Appendix

| Appendix | Content |
|----------|---------|
| [A · Glossary](./appendix-a-glossary) | 60+ terms across 7 categories |
| [B · Struct Reference](./appendix-b-structs) | All C structs in one place |

---

## ❓ FAQ

**Q: What background do I need?**
Basic C syntax (pointers, structs, for loops). No deep learning knowledge required.

**Q: What hardware?**
Any computer that can compile C. 2GB RAM (988MB model + fp32 conversion).

**Q: Do I need PyTorch?**
No. The engine is pure C. PyTorch is only used in optional verification (`verify.py`).

**Q: Can I use this in production?**
No. ~3 tok/s (1000× slower than vLLM), no concurrency. This is a learning project. See [Chapter 10](./10-ecosystem).
