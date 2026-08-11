<div align="center">

# inference-learn

**从零用纯 C 写一个 Transformer 推理引擎，理解 LLM 的每一个零件**

Build a Transformer inference engine from scratch in pure C — no PyTorch, no AI frameworks, just libc.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Language: C](https://img.shields.io/badge/Language-C99-blue.svg)]()
[![Model: Qwen2.5-0.5B](https://img.shields.io/badge/Model-Qwen2.5--0.5B-green.svg)]()
[![Lines: ~2700](https://img.shields.io/badge/Code-~2700行_C-orange.svg)]()
[![Docs: 中/EN](https://img.shields.io/badge/Docs-中文/English-red.svg)]()

📖 **[在线文档 / Docs](https://jackiesre721.github.io/inference-learn/cn/)** · 🇬🇧 **[English Docs](https://jackiesre721.github.io/inference-learn/en/)** · 📦 **[Source](https://github.com/jackiesre721/inference-learn)**

</div>

> A minimal LLM inference engine written in pure C (~2700 lines, zero dependencies).
> Loads Qwen2.5-0B weights (safetensors), implements RMSNorm / RoPE / GQA / SwiGLU / KV Cache / BPE from scratch.
> Numerically verified against PyTorch (error < 0.0002).
> Full bilingual documentation: [中文](https://jackiesre721.github.io/inference-learn/cn/) | [English](https://jackiesre721.github.io/inference-learn/en/)

---

## ✨ Demo

```bash
$ ./run model.safetensors tokenizer.bin "1+1="
1+1=2，2+2=4，3+3=6，4+4=8，5+5=10，6+6=12，7+7=14，8+8=16...

$ ./run model.safetensors tokenizer.bin "The meaning of life is"
The meaning of life is a question that has been debated for centuries.
Some people believe that life is a journey of self-discovery and growth...

$ ./run model.safetensors tokenizer.bin "Python is"
Python is a programming language that is widely used for web development...
```

**这不是调 API** — This is not an API call. It's a ~2700-line pure C inference engine that hand-writes every operator (RMSNorm, RoPE, GQA, SwiGLU, KV Cache, BPE tokenizer), loads a real 4.94-billion-parameter model, and generates text on CPU.

## 🎯 What is this / 这是什么

A pure-C inference engine that loads [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) (0.49B params) and generates text.

**Goal is not performance** (1000× slower than vLLM), but **understanding every component**:

- How weights are organized on disk / 权重文件在磁盘上长什么样
- What happens inside 24 Transformer layers when a token enters / 一个 token 进来后发生了什么
- Why RMSNorm, RoPE, GQA, SwiGLU, KV Cache / 为什么用这些技术
- Where the model's "intelligence" lives in 500M numbers / 模型的"智能"藏在哪

Every operator is a hand-written triple loop, no black boxes. **Forward pass verified against PyTorch, error < 0.0002**.

## 🚀 Quick Start

```bash
# 1. Download model weights (~1GB)
./download.sh

# 2. Export tokenizer
python3 export_tokenizer.py

# 3. Build
make

# 4. Generate text
./run model.safetensors tokenizer.bin "Python is"

# 5. Visualize internals (HTML report)
./run model.safetensors tokenizer.bin "Hello" --report
open trace_report.html
```

<details>
<summary><b>📋 All commands / 所有命令</b></summary>

| Command | What it does |
|---------|-------------|
| `./run model.safetensors tokenizer.bin "prompt"` | Generate text / 生成文本 |
| `./run model.safetensors tokenizer.bin "..." -v` | Debug log: per-layer summary |
| `./run model.safetensors tokenizer.bin "..." -vv` | Trace log: all intermediate tensors |
| `./run model.safetensors tokenizer.bin "..." --report` | Generate HTML visualization report |
| `./run model.safetensors tokenizer.bin "..." -t 0.8 -k 40` | Temperature + top-k sampling |
| `./run info model.safetensors` | Inspect model structure and all tensors |
| `./run encode tokenizer.bin "Hello world"` | See BPE tokenization process |
| `./run fwd model.safetensors 198 0` | Single-token forward (numerical verification) |

</details>

## 🏗️ Architecture / 系统架构图

```mermaid
flowchart TB
    subgraph Input["📥 输入 Input"]
        PROMPT["用户输入<br/>'Hello world'"]
    end

    subgraph Tokenize["🔤 分词 tokenizer.c"]
        BPE["BPE 编码<br/>文字 → token ids<br/>Hello world → 9707, 1879"]
    end

    subgraph Loading["📦 权重加载 safetensors.c"]
        FILE["model.safetensors<br/>(988MB, bf16)"]
        MMAP["mmap 零拷贝<br/>bf16 → fp32 转换"]
        FILE --> MMAP
    end

    subgraph Engine["⚡ 推理引擎 net.c ~400行"]
        EMBED["Embedding 查表<br/>token id → 896维向量"]
        subgraph Layer["× 24 层 Transformer"]
            NORM1["RMSNorm 归一化"]
            QKV["QKV 投影<br/>算出 Query / Key / Value"]
            ROPE["RoPE 旋转位置编码"]
            ATT["Attention 注意力<br/>GQA: 14个Q头共享2组KV"]
            CACHE["KV Cache 缓存"]
            NORM2["RMSNorm 归一化"]
            MLP["SwiGLU MLP 前馈<br/>扩张4864维 → 门控 → 压回"]
            NORM1 --> QKV --> ROPE --> ATT --> CACHE --> NORM2 --> MLP
        end
        LOGITS["Logits 151936个分数<br/>和 embedding 表做点积"]
        EMBED --> Layer --> LOGITS
    end

    subgraph Output["📤 输出 run.c"]
        SAMPLE["采样<br/>greedy / temperature / top-k"]
        DECODE["BPE 解码<br/>token id → 文字"]
        PRINT["输出文本"]
        SAMPLE --> DECODE --> PRINT
    end

    PROMPT --> BPE
    BPE --> EMBED
    MMAP --> EMBED
    LOGITS --> SAMPLE
```

## 📂 Source Structure

```
inference-learn/
├── run.c              # Entry: generation loop + sampling (~500 lines)
├── net.c              # Core: forward pass, all operators (~400 lines) ← read this first
├── tokenizer.c        # BPE tokenizer (~500 lines)
├── safetensors.c      # Weight loading: mmap + bf16→fp32 (~300 lines)
├── json.c             # Minimal JSON parser (~280 lines)
├── trace.c            # Tiered logging system (~210 lines)
├── report.c           # HTML report generator (~150 lines)
├── docs/              # ★ Full documentation (bilingual: cn/ + en/)
│   ├── cn/            # 中文文档 (14 files)
│   └── en/            # English docs (14 files)
└── *.py               # Helper scripts (tokenizer export, verification)
```

## 📖 Documentation / 学习文档

**📖 [中文文档](https://jackiesre721.github.io/inference-learn/cn/)** · **🇬🇧 [English Docs](https://jackiesre721.github.io/inference-learn/en/)**

10 chapters + 2 appendices, 13 Mermaid diagrams, bilingual (Chinese + English):

| Chapter | Topic | Source |
|---------|-------|--------|
| [1. Basics](https://jackiesre721.github.io/inference-learn/cn/01-basics) | Tensors, dimensions, how 0.5B is calculated | — |
| [2. Weights](https://jackiesre721.github.io/inference-learn/cn/02-weights) | safetensors format, mmap, bf16 | `safetensors.c` |
| [3. JSON Parser](https://jackiesre721.github.io/inference-learn/cn/03-json) | Recursive descent parser | `json.c` |
| [4. Model Architecture](https://jackiesre721.github.io/inference-learn/cn/04-model) | Config / Weights / RunState structs | `net.h` |
| [5. Forward Pass ★](https://jackiesre721.github.io/inference-learn/cn/05-forward) | **Core**: all 6 operators + 6 diagrams | `net.c` |
| [6. Tokenizer](https://jackiesre721.github.io/inference-learn/cn/06-tokenizer) | BPE, byte-level, pre-tokenize | `tokenizer.c` |
| [7. Sampling](https://jackiesre721.github.io/inference-learn/cn/07-sampling) | temperature, top-k, prefill/decode | `run.c` |
| [8. Logging](https://jackiesre721.github.io/inference-learn/cn/08-logging) | Tiered log, HTML report | `trace.c` |
| [9. Debugging](https://jackiesre721.github.io/inference-learn/cn/09-debugging) | Numerical verification, ASan | — |
| [10. Ecosystem](https://jackiesre721.github.io/inference-learn/cn/10-ecosystem) | vs llama.cpp / vLLM / SGLang | — |

## 🧠 What you'll learn / 你会学到什么

- ✅ How 500M weights are organized, stored, and loaded / 5 亿个权重怎么存
- ✅ What happens step-by-step inside 24 Transformer layers / 24 层内部发生了什么
- ✅ **RMSNorm**（归一化）: 不用它数值会爆炸，为什么比 LayerNorm 好
- ✅ **RoPE**（旋转位置编码）: 用旋转角度编码位置，为什么比绝对位置编码强
- ✅ **GQA**（分组注意力）: 14 个头共享 2 组 KV，省 7 倍内存
- ✅ **KV Cache**（键值缓存）: 把生成复杂度从 O(N²) 降到 O(N)
- ✅ **BPE**（分词器）: 怎么把文字切成 token，采样怎么选词
- ✅ How your engine compares to vLLM — where, how much, why / 和 vLLM 差在哪
- ✅ Why LLMs use GPUs not CPUs (the 100-1000× gap) / 为什么用 GPU

**Out of scope / 不涉及**: training, backpropagation, GPU programming, distributed deployment / 训练、反向传播、GPU 编程、分布式部署

<details>
<summary><b>🔤 核心术语速查（看不懂缩写？点这里）</b></summary>

| 术语 | 全称 | 一句话解释 |
|------|------|-----------|
| **RMSNorm** | Root Mean Square Normalization | 归一化。防止数值经过 24 层后爆炸。每层做 2 次 |
| **LayerNorm** | Layer Normalization | 归一化的老版本（多一步减均值）。现代模型用 RMSNorm 替代 |
| **RoPE** | Rotary Position Embedding | 旋转位置编码。用旋转角度告诉模型"这个词在第几个位置" |
| **GQA** | Grouped Query Attention | 分组注意力。14 个查询头共享 2 组 KV，省 7 倍内存 |
| **SwiGLU** | Swish-Gated Linear Unit | 门控激活函数。决定哪些信息通道该激活、哪些该抑制 |
| **KV Cache** | Key-Value Cache | 键值缓存。把算过的 K/V 存起来避免重算，生成速度 ×100 |
| **BPE** | Byte-Pair Encoding | 字节对编码分词。把文字切成 token（如 "Hello" → 9707）|
| **Attention** | 注意力机制 | 让每个词"看到"其他词，决定关注谁、忽略谁 |
| **Token** | — | 模型处理的最小单位。一个中文字或一个英文片段 |
| **Logits** | — | 模型对每个词打的分。分数最高的就是最可能的下一个词 |
| **Embedding** | 词嵌入 | 查表把 token 变成 896 个数字，是模型"理解"词的方式 |
| **Prefill** | 预填充 | 把 prompt 逐个读进 KV Cache，不采样 |
| **Decode** | 解码 | 用 KV Cache 逐个生成新 token |

</details>

## 📊 Performance

| Metric | Value | vs vLLM |
|--------|-------|---------|
| Generation speed | ~3 tokens/s | vLLM ~3000 tok/s (1000× slower) |
| Memory | ~2 GB | — |
| Model file | 988 MB (bf16) | — |
| Numerical accuracy | error < 0.0002 vs PyTorch | — |
| Code size | ~2700 lines C, 0 deps | vLLM ~200K lines, 30+ deps |

Slow because: pure fp32 + CPU serial + no quantization. This is by **design** — simplicity over performance. See [Chapter 10](https://jackiesre721.github.io/inference-learn/cn/10-ecosystem).

## ✅ Verified

- [x] Forward pass numerically aligned with PyTorch (error < 0.0002)
- [x] Greedy generation matches HF `model.generate()` token-by-token
- [x] BPE encoding matches HF tokenizer (English)
- [x] ASan/UBSan clean

## 🙏 Acknowledgments

- [Andrej Karpathy / llama2.c](https://github.com/karpathy/llama2.c) — "minimal learnable inference engine" philosophy
- [Qwen Team / Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) — open model weights
- [HuggingFace / safetensors](https://github.com/huggingface/safetensors) — clean weight format

## 📄 License

MIT

---

<details>
<summary><b>🏷️ Keywords</b></summary>

**English**: llm inference engine | transformer from scratch | pure c | qwen2.5 | gqa grouped query attention | rope rotary position embedding | rmsnorm | swiglu | kv cache | bpe tokenizer | safetensors | deep learning | language model | nano inference | learn llm | c99

**中文**: 大模型推理引擎 | 从零实现 transformer | 纯 c 大模型 | qwen2.5 | gqa 分组注意力 | rope 旋转位置编码 | rmsnorm | swiglu | kv cache | bpe 分词器 | safetensors | 深度学习 | 语言模型 | llm 学习 | 推理引擎原理

</details>
