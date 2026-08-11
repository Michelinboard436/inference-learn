# 学习文档

> 本目录是 `inference-learn` 项目的完整学习文档。
> 按章节顺序读，每章配合对应的源码文件。

## 目录

| 章节 | 文件 | 主题 | 配合源码 |
|------|------|------|---------|
| [第 1 章](./01-basics.md) | `01-basics.md` | 基础概念：什么是推理引擎、张量与维度 | — |
| [第 2 章](./02-weights.md) | `02-weights.md` | 权重的存储与加载 | `safetensors.c` |
| [第 3 章](./03-json.md) | `03-json.md` | JSON 解析器 | `json.c` |
| [第 4 章](./04-model.md) | `04-model.md` | 模型结构与三大结构体 | `net.h` |
| [第 5 章](./05-forward.md) | `05-forward.md` | 前向传播（核心，最长） | `net.c` |
| [第 6 章](./06-tokenizer.md) | `06-tokenizer.md` | 分词器 BPE | `tokenizer.c` |
| [第 7 章](./07-sampling.md) | `07-sampling.md` | 采样与生成 | `run.c` |
| [第 8 章](./08-logging.md) | `08-logging.md` | 日志与可视化 | `trace.c` |
| [第 9 章](./09-debugging.md) | `09-debugging.md` | 调试与验证方法论 | — |
| [第 10 章](./10-ecosystem.md) | `10-ecosystem.md` | 生态对比与性能优化 | — |
| [附录 A](./appendix-a-glossary.md) | `appendix-a-glossary.md` | 术语总表 | — |
| [附录 B](./appendix-b-structs.md) | `appendix-b-structs.md` | 结构体速查 | — |

## 建议阅读顺序

**新手**：1 → 2 → 4 → 5 → 6 → 7（核心路径，约 2-3 小时）

**想深入**：1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10（完整路径，约 1 周）

**只关心原理**：4 → 5（模型结构 + 前向传播）

**想对比工业引擎**：直接看第 10 章
