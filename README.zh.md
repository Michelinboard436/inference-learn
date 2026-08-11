<div align="center">

# inference-learn

**从零用纯 C 写一个 Transformer 推理引擎，理解 LLM 的每一个零件**

不依赖 PyTorch，不依赖任何 AI 框架，只靠 libc。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Language: C](https://img.shields.io/badge/Language-C99-blue.svg)]()
[![Model: Qwen2.5-0.5B](https://img.shields.io/badge/Model-Qwen2.5--0.5B-green.svg)]()
[![Code: ~2700](https://img.shields.io/badge/Code-~2700行-orange.svg)]()

📖 **[在线文档](https://jackiesre721.github.io/inference-learn/)** · [English](./README.md) · [源码](https://github.com/jackiesre721/inference-learn)

</div>

---

## ✨ 演示

```bash
$ ./run model.safetensors tokenizer.bin "1+1="
1+1=2，2+2=4，3+3=6，4+4=8，5+5=10，6+6=12，7+7=14，8+8=16...

$ ./run model.safetensors tokenizer.bin "The meaning of life is"
The meaning of life is a question that has been debated for centuries.
Some people believe that life is a journey of self-discovery and growth...

$ ./run model.safetensors tokenizer.bin "Python is"
Python is a programming language that is widely used for web development...
```

这不是调 API，是用 ~2700 行纯 C 从零实现的推理引擎。手写了 RMSNorm、RoPE、GQA、SwiGLU、KV Cache、BPE 分词器，加载真实的 4.94 亿参数大模型，在 CPU 上生成文本。

## 🎯 这是什么

一个用纯 C 写的推理引擎，加载 [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)（0.49B 参数）并生成文本。

目标不是性能（比 vLLM 慢 1000 倍），而是让你看懂推理引擎的每个零件：

- 权重文件在磁盘上长什么样
- 一个 token 进来后，24 层 Transformer 内部一步步发生了什么
- 为什么用 RMSNorm、RoPE、GQA、SwiGLU、KV Cache
- 模型的"智能"到底藏在 5 亿个数字的哪里

每个算子都是手写的三重循环，没有黑盒。前向传播数值和 PyTorch 官方实现对比，误差 < 0.0002。

## 🚀 快速开始

```bash
./download.sh            # 下载权重 (~1GB)
python3 export_tokenizer.py  # 导出分词器
make                     # 编译
./run model.safetensors tokenizer.bin "Python is"
```

<details>
<summary><b>📋 所有命令</b></summary>

| 命令 | 作用 |
|------|------|
| `./run model.safetensors tokenizer.bin "提示词"` | 生成文本 |
| `./run model.safetensors tokenizer.bin "..." -v` | debug 日志：每层摘要 |
| `./run model.safetensors tokenizer.bin "..." -vv` | trace 日志：所有中间张量 |
| `./run model.safetensors tokenizer.bin "..." --report` | 生成 HTML 可视化报告 |
| `./run model.safetensors tokenizer.bin "..." -t 0.8 -k 40` | 温度 + top-k 采样 |
| `./run info model.safetensors` | 查看模型结构 |
| `./run encode tokenizer.bin "Hello world"` | 查看 BPE 分词过程 |
| `./run fwd model.safetensors 198 0` | 单 token 前向（数值验证） |

</details>

## 📖 学习文档

**[在线文档](https://jackiesre721.github.io/inference-learn/)**：10 章 + 2 附录，13 张 Mermaid 图

| 章 | 主题 | 配合源码 |
|----|------|---------|
| [1. 基础概念](https://jackiesre721.github.io/inference-learn/01-basics) | 张量、维度、0.5B 怎么算 | — |
| [2. 权重加载](https://jackiesre721.github.io/inference-learn/02-weights) | safetensors、mmap、bf16 | `safetensors.c` |
| [3. JSON 解析](https://jackiesre721.github.io/inference-learn/03-json) | 递归下降解析器 | `json.c` |
| [4. 模型结构](https://jackiesre721.github.io/inference-learn/04-model) | 三大结构体 | `net.h` |
| [5. 前向传播 ★](https://jackiesre721.github.io/inference-learn/05-forward) | 6 个算子 + 6 张图 | `net.c` |
| [6. 分词器](https://jackiesre721.github.io/inference-learn/06-tokenizer) | BPE、byte-level | `tokenizer.c` |
| [7. 采样生成](https://jackiesre721.github.io/inference-learn/07-sampling) | temperature、top-k、prefill/decode | `run.c` |
| [8. 日志可视化](https://jackiesre721.github.io/inference-learn/08-logging) | 分级日志、HTML 报告 | `trace.c` |
| [9. 调试验证](https://jackiesre721.github.io/inference-learn/09-debugging) | 数值验证、ASan | — |
| [10. 生态对比](https://jackiesre721.github.io/inference-learn/10-ecosystem) | vs llama.cpp / vLLM / SGLang | — |

## 🧠 你会学到什么

- 5 亿个权重数字怎么组织、存储、加载
- 输入一句话后，模型内部一步步发生了什么
- RMSNorm（归一化）：不用它数值会爆炸，为什么比 LayerNorm 好
- RoPE（旋转位置编码）：用旋转角度编码位置，为什么比绝对位置编码强
- GQA（分组注意力）：14 个头共享 2 组 KV，省 7 倍内存
- KV Cache（键值缓存）：把生成复杂度从 O(N²) 降到 O(N)
- BPE（分词器）：怎么把文字切成 token，采样怎么选词
- 你的引擎和 vLLM 差在哪、差多少、为什么
- 为什么大模型用 GPU 不用 CPU

不涉及：训练、反向传播、GPU 编程、分布式部署。

## 📊 性能

| 指标 | 值 | 对比 vLLM |
|------|-----|-----------|
| 生成速度 | ~3 tokens/s | vLLM ~3000 tok/s（慢 1000 倍） |
| 内存 | ~2 GB | — |
| 数值精度 | 误差 < 0.0002 | — |
| 代码量 | ~2700 行，0 依赖 | vLLM ~20 万行，30+ 依赖 |

## ✅ 已验证

- [x] 前向传播数值和 PyTorch 对齐（误差 < 0.0002）
- [x] 贪心生成和 HF 逐 token 一致
- [x] BPE 编码和 HF tokenizer 一致
- [x] ASan/UBSan 全部通过

## 🙏 致谢

- [llama2.c](https://github.com/karpathy/llama2.c) — 最小可学习推理引擎理念
- [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) — 开放模型权重
- [safetensors](https://github.com/huggingface/safetensors) — 权重格式

## 📄 License

MIT
