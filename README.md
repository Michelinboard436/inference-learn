<div align="center">

# inference-learn

**从零用纯 C 写一个 Transformer 推理引擎，理解 LLM 的每一个零件**

不依赖 PyTorch，不依赖任何 AI 框架，只靠 libc。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Language: C](https://img.shields.io/badge/Language-C99-blue.svg)]()
[![Model: Qwen2.5-0.5B](https://img.shields.io/badge/Model-Qwen2.5--0.5B-green.svg)]()
[![Lines: ~2700](https://img.shields.io/badge/Code-~2700行_C-orange.svg)]()
[![Docs: 中文](https://img.shields.io/badge/Docs-中文-red.svg)]()

</div>

---

## ✨ 30 秒演示

```bash
$ ./run model.safetensors tokenizer.bin "1+1="
1+1=2，2+2=4，3+3=6，4+4=8，5+5=10，6+6=12，7+7=14，8+8=16...

$ ./run model.safetensors tokenizer.bin "The meaning of life is"
The meaning of life is a question that has been debated for centuries.
Some people believe that life is a journey of self-discovery and growth...

$ ./run model.safetensors tokenizer.bin "Python is"
Python is a programming language that is widely used for web development...
```

**这不是调 API，是用 ~2700 行纯 C 从零实现的推理引擎**——手写了 RMSNorm、RoPE、GQA、SwiGLU、KV Cache、BPE 分词器，加载真实的 4.94 亿参数大模型，在 CPU 上生成文本。

## 🎯 这是什么

一个能加载 [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)（0.49B 参数）并生成文本的推理引擎。

**目标不是性能**（比 vLLM 慢 1000 倍），而是**让你看懂推理引擎的每个零件**：

- 权重文件在磁盘上长什么样
- 一个 token 进来后，24 层 Transformer 内部一步步发生了什么
- 为什么用 RMSNorm、RoPE、GQA、SwiGLU、KV Cache
- 模型的"智能"到底藏在 5 亿个数字的哪里

每个算子都是手写的三重循环，没有黑盒。**前向传播数值和 PyTorch 官方实现对比，误差 < 0.0002**。

## 🚀 快速开始

```bash
# 1. 下载模型权重 (~1GB)
./download.sh

# 2. 导出分词器
python3 export_tokenizer.py

# 3. 编译
make

# 4. 生成文本
./run model.safetensors tokenizer.bin "Python is"

# 5. 看引擎内部每一步 (HTML 报告)
./run model.safetensors tokenizer.bin "Hello" --report
open trace_report.html
```

<details>
<summary><b>📋 所有命令</b></summary>

| 命令 | 作用 |
|------|------|
| `./run model.safetensors tokenizer.bin "你的提示"` | 生成文本 |
| `./run model.safetensors tokenizer.bin "..." -v` | debug 日志：每层摘要 |
| `./run model.safetensors tokenizer.bin "..." -vv` | trace 日志：所有中间张量 |
| `./run model.safetensors tokenizer.bin "..." --report` | 生成 HTML 可视化报告 |
| `./run model.safetensors tokenizer.bin "..." -t 0.8 -k 40` | 带温度和 top-k 采样 |
| `./run info model.safetensors` | 查看模型结构和所有张量 |
| `./run encode tokenizer.bin "Hello world"` | 看 BPE 分词过程 |
| `./run fwd model.safetensors 198 0` | 单 token 前向（数值验证用） |

</details>

## 📂 源码结构

```
inference-learn/
├── run.c              # 入口：生成循环 + 采样 (~500 行)
├── net.c              # 核心：前向传播全部算子 (~400 行) ← 重点读这个
├── tokenizer.c        # BPE 分词器 (~500 行)
├── safetensors.c      # 权重加载：mmap + bf16→fp32 (~300 行)
├── json.c             # 极简 JSON 解析器 (~280 行)
├── trace.c            # 分级日志系统 (~210 行)
├── report.c           # HTML 报告生成 (~150 行)
├── docs/              # ★ 完整学习文档 (10 章, 中文)
└── *.py               # 辅助脚本（分词器导出、数值验证）
```

## 📖 学习文档

完整的中文学习文档在 [`docs/`](./docs/) 目录，**10 章 + 2 附录，6700+ 行，13 个 Mermaid 可视化图**：

| 章 | 主题 | 配合源码 |
|----|------|---------|
| [1. 基础概念](./docs/01-basics.md) | 张量、维度、0.5B 怎么算 | — |
| [2. 权重加载](./docs/02-weights.md) | safetensors 格式、mmap、bf16 | `safetensors.c` |
| [3. JSON 解析](./docs/03-json.md) | 递归下降解析器 | `json.c` |
| [4. 模型结构](./docs/04-model.md) | Config/Weights/RunState 三大结构体 | `net.h` |
| [5. 前向传播](./docs/05-forward.md) | **核心**：6 个算子逐个拆解 + 6 张图 | `net.c` |
| [6. 分词器](./docs/06-tokenizer.md) | BPE、byte-level、pre-tokenize | `tokenizer.c` |
| [7. 采样生成](./docs/07-sampling.md) | temperature、top-k、prefill/decode | `run.c` |
| [8. 日志可视化](./docs/08-logging.md) | 分级日志、HTML 报告 | `trace.c` |
| [9. 调试验证](./docs/09-debugging.md) | 数值验证、ASan、逐层 dump | — |
| [10. 生态对比](./docs/10-ecosystem.md) | vs llama.cpp / vLLM / SGLang / 长上下文 | — |

📖 [完整的阅读路径和目录 →](./docs/README.md)

## 🧠 你会学到什么

读完这个项目，你会理解：

- ✅ 5 亿个权重数字是怎么组织、存储、加载的
- ✅ 输入一句话后，模型内部一步步发生了什么计算
- ✅ 为什么用 RMSNorm（不归一化会爆炸）、RoPE（旋转编码位置）、GQA（省 7 倍内存）
- ✅ KV Cache 怎么把生成复杂度从 O(N²) 降到 O(N)
- ✅ BPE 怎么把文字切成 token、采样怎么从概率分布选词
- ✅ 你的手写引擎和 vLLM 差在哪、差多少、为什么
- ✅ 为什么大模型用 GPU 不用 CPU（100-1000 倍差距的根源）

**不会**教你（超出范围）：训练、反向传播、GPU 编程、分布式部署。

## 📊 性能

| 指标 | 值 | 对比 vLLM |
|------|-----|-----------|
| 生成速度 | ~3 tokens/s | vLLM ~3000 tok/s（慢 1000 倍） |
| 内存占用 | ~2 GB | — |
| 模型文件 | 988 MB（bf16） | — |
| 数值精度 | 与 PyTorch 误差 < 0.0002 | — |
| 代码总量 | ~2700 行 C + 0 依赖 | vLLM ~20 万行 + 30 依赖 |

慢是因为纯 fp32 + CPU 串行 + 无量化。这是**设计目标**——简洁优先，不是性能优先。详见 [第 10 章](./docs/10-ecosystem.md)。

## 🔧 技术栈

| 项目 | 选择 |
|------|------|
| 语言 | C99，只依赖 libc |
| 模型 | Qwen2.5-0.5B（Llama 风格 + QKV bias + GQA） |
| 权重格式 | 直接读 safetensors（mmap + bf16→fp32） |
| 分词器 | C 手写 BPE（byte-level） |
| 构建 | `make`（单条 gcc 命令） |
| 依赖 | **0**（只要 libc） |

## ✅ 已验证

- [x] 前向传播数值和 PyTorch 对齐（误差 < 0.0002）
- [x] 贪心生成和 HF `model.generate()` 逐 token 一致
- [x] BPE 编码和 HF tokenizer 一致（英文场景）
- [x] ASan/UBSan 全部通过

## 🙏 致谢

- [Andrej Karpathy / llama2.c](https://github.com/karpathy/llama2.c) — "最小可学习推理引擎"的理念
- [Qwen 团队 / Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) — 开放的模型权重
- [HuggingFace / safetensors](https://github.com/huggingface/safetensors) — 简洁的权重格式

## 📄 License

MIT
