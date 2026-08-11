# 📖 inference-learn 目录

> 一本从零理解 LLM 推理引擎的电子书。~2700 行 C 代码，0 依赖，加载 Qwen2.5-0.5B。

<div align="center">

| | |
|:---:|:---:|
| **代码** | ~2,700 行 C99 |
| **依赖** | 0（只要 libc） |
| **文档** | 10 章 + 2 附录 |
| **模型** | Qwen2.5-0.5B（0.49B 参数） |
| **精度** | 与 PyTorch 误差 < 0.0002 |

</div>

---

## 🗺️ 怎么读这本书

三种阅读路径，选适合你的：

<table>
<tr>
<td width="33%" align="center">

### 🚀 快速入门

**2-3 小时**

</td>
<td width="33%" align="center">

### 📚 完整学习

**约 1 周**

</td>
<td width="33%" align="center">

### 🔬 直奔核心

**1-2 小时**

</td>
</tr>
<tr>
<td>

→ [1 基础](./01-basics)
→ [2 权重](./02-weights)
→ [4 模型](./04-model)
→ [5 前向传播](./05-forward)
→ [6 分词器](./06-tokenizer)
→ [7 采样](./07-sampling)

</td>
<td>

→ 从第 1 章到第 10 章
→ 逐章读完
→ 配合源码理解每个算子
→ [从第 1 章开始](./01-basics)

</td>
<td>

→ 只看前向传播原理
→ [4 模型结构](./04-model)
→ [5 前向传播 ★](./05-forward)
→ 最核心的两章
→ 891 行，最长的一章

</td>
</tr>
</table>

---

## Part 1 · 基础

> 万丈高楼平地起——先搞清楚权重文件长什么样、维度代表什么。

### [第 1 章 · 基础概念](./01-basics)

什么是推理引擎、张量与维度、「0.5B」怎么算出来的、每一层有哪些参数。

> 配合源码：全局 · 403 行

### [第 2 章 · 权重的存储与加载](./02-weights)

safetensors 文件格式、mmap 零拷贝加载、bf16→fp32 位转换、embedding 表的精确位置。

> 配合源码：`safetensors.c` · 320 行

### [第 3 章 · JSON 解析器](./03-json)

为什么自己写 JSON 解析器、递归下降算法、JsonValue 树结构。

> 配合源码：`json.c` · 328 行

---

## Part 2 · 核心

> 引擎的心脏——模型结构和前向传播的全部算子。

### [第 4 章 · 模型结构](./04-model)

Config / TransformerWeights / RunState 三大结构体，net.h 里每一个字段的含义。

> 配合源码：`net.h` · 769 行

### [第 5 章 · 前向传播 ★](./05-forward)

**最重要的一章。** RMSNorm、matmul、RoPE、Attention、GQA、SwiGLU、KV Cache、残差连接——6 个算子逐个拆解，含 cat dog 真实数值演示 + 多 token 协同。

> 配合源码：`net.c` · 891 行 · 6 张 Mermaid 图

### [第 6 章 · 分词器 BPE](./06-tokenizer)

文字怎么变成 token、BPE 合并算法、byte-level 映射、pre-tokenization。

> 配合源码：`tokenizer.c` · 487 行

### [第 7 章 · 采样与生成](./07-sampling)

prefill + decode 两阶段、temperature / top-k 采样、轮盘赌、为什么 prefill 计算密集 decode 内存密集。

> 配合源码：`run.c` · 443 行

---

## Part 3 · 进阶

> 从「能跑」到「能调试、能对比、能优化」。

### [第 8 章 · 日志与可视化](./08-logging)

分级日志系统（-v / -vv）、彩色输出、HTML 可视化报告。

> 配合源码：`trace.c` · 487 行

### [第 9 章 · 调试与验证方法论](./09-debugging)

为什么必须数值验证、单 token 验证法、ASan 抓内存 bug、逐层 dump 定位。

> 470 行

### [第 10 章 · 生态对比与性能优化](./10-ecosystem)

你的引擎和 llama.cpp / vLLM / SGLang 差在哪、差多少、为什么。GPU vs CPU、量化、大模型加载、长上下文。

> 1065 行 · 最长的章

---

## 📚 附录

| 附录 | 内容 | 行数 |
|------|------|------|
| [附录 A · 术语总表](./appendix-a-glossary) | 7 大类 60+ 术语速查 | 229 行 |
| [附录 B · 结构体速查](./appendix-b-structs) | 所有文件的 C 结构体定义 | 114 行 |

---

## ❓ 常见问题

**Q: 需要什么基础？**
能看懂 C 语言基本语法（指针、struct、for 循环）。不要求懂深度学习。

**Q: 需要什么硬件？**
任何能跑 C 程序的电脑。2GB 内存即可（模型权重 988MB + fp32 转换）。

**Q: 需要装 PyTorch 吗？**
不需要。引擎完全用 C 写，PyTorch 只在数值验证（`verify.py`）时用到，且是可选的。

**Q: 比 llama2.c 强在哪？**
目标模型不同（Qwen2.5 vs Llama 2）、文件格式不同（直接读 safetensors vs 自定义 bin）、架构适配不同（QKV bias + GQA）。详见 [README](https://github.com/jackiesre721/inference-learn)。

**Q: 能用来做生产部署吗？**
不能。速度 ~3 tok/s（比 vLLM 慢 1000 倍），不支持并发。这是学习项目，不是生产工具。详见[第 10 章](./10-ecosystem)。
