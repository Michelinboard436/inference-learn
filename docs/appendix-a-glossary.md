# 附录 A 术语总表

> 附录 A | [上一章：第 10 章 生态对比](./10-ecosystem.md) | [下一章：附录 B 结构体速查](./appendix-b-structs.md)

> 这份术语表把全书（第 1-10 章）出现的术语按类别整理。
> 读其他章节遇到不懂的词，来这里查。
> 每个术语都标了**英文 / 全称**，方便对照 HuggingFace 文档和论文。

---

## 目录

- [A.1 模型结构术语](#a1-模型结构术语)
- [A.2 注意力术语](#a2-注意力术语)
- [A.3 归一化与激活术语](#a3-归一化与激活术语)
- [A.4 分词术语](#a4-分词术语)
- [A.5 数据格式术语](#a5-数据格式术语)
- [A.6 C 语言 / 工程术语](#a6-c-语言--工程术语)
- [A.7 生态与性能术语](#a7-生态与性能术语)

---

## A.1 模型结构术语

| 术语 | 英文 | 含义 |
|------|------|------|
| 残差流宽度 | hidden size / dim | token 向量长度（Qwen2.5-0.5B 是 896） |
| MLP 中间宽度 | intermediate size | 前馈网络中间层（4864） |
| 层数 | num layers | Transformer 重复次数（24） |
| 头 | head | 注意力的并行单元 |
| 头维度 | head dim | 每个头的向量长度（64） |
| 词表大小 | vocab size | 认识多少 token（151936） |
| 序列长度 | seq len | 最大 token 数（2048） |
| 词嵌入 | embedding | token → 向量的查找表 |
| 投影 | projection | 矩阵乘法 `y = Wx + b` |
| 偏置 | bias | 投影后加的常数向量 |
| 残差连接 | residual | `x = x + layer(x)`，让信息能跳过某些层 |
| 前馈网络 | FFN / MLP | 每层里「独立思考」的子模块 |
| 解码器 | decoder | 从左往右逐词生成的 Transformer 部分 |
| 自回归 | autoregressive | 输出又变成输入，循环生成 |
| 参数量 | parameters | 所有权重矩阵的元素总数（0.5B ≈ 4.94 亿） |
| 张量 | tensor | 多维数组 |
| 标量 | scalar | 0 维张量（单个数） |
| 向量 | vector | 1 维张量（一排数） |
| 矩阵 | matrix | 2 维张量（行列网格） |
| 特征维 | feature dim | 每个 token 向量的列数（896），与行数（seq_len）无关 |

---

## A.2 注意力术语

| 术语 | 英文 | 含义 |
|------|------|------|
| Query | Q | 当前 token「找什么」的向量 |
| Key | K | 每个 token「是什么」的向量 |
| Value | V | 每个 token 的内容向量 |
| 注意力 | attention | 让每个词看其他词，决定关注谁 |
| 注意力分数 | attention score | Q·Kᵀ 算出的相关程度 |
| 缩放因子 | scale（√d） | 防止点积数值过大，除以 √head_dim |
| softmax | — | 把分数归一化成概率（和为 1） |
| 因果注意力 | causal attention | 只看左边不看右边（`for t<=pos`） |
| GQA | Grouped Query Attention | 多个 Q 头共享 KV（Qwen 是 14 个 Q 头共享 2 个 KV 头） |
| KV Cache | Key-Value Cache | 缓存历史 K/V 避免重算，把生成复杂度从 O(N²) 降到 O(N) |
| RoPE | Rotary Position Embedding | 旋转位置编码，用旋转编码位置信息 |
| rope_theta | RoPE base frequency | 基频，越大能编码的位置越远（Qwen 是 1e6） |
| 点积 | dot product | 两个向量对应位相乘再求和，衡量方向一致性 |
| 因果性 | causality | 当前 token 只能看到之前的 token |

---

## A.3 归一化与激活术语

| 术语 | 英文 | 含义 |
|------|------|------|
| 归一化 | normalization | 把向量拉回标准大小，防止多层累积爆炸 |
| RMSNorm | Root Mean Square Norm | 除以均方根（比 LayerNorm 少一步减均值） |
| LayerNorm | Layer Normalization | 减均值 → 除标准差 → 乘 weight（三步） |
| eps | epsilon | 防除零的小常数（1e-6） |
| 均方根 | RMS | Root Mean Square，`√(Σx²/n)` |
| SwiGLU | Swish-Gated Linear Unit | 门控前馈网络，`silu(gate) × up` |
| silu | Sigmoid Linear Unit | `x · sigmoid(x)`，比 ReLU 平滑 |
| gate | 门控 | 动态决定哪些通道激活（软门，0 到 1） |
| ReLU | Rectified Linear Unit | `max(0, x)`，硬门（要么 0 要么原值） |
| 激活函数 | activation function | 引入非线性的函数 |
| 激活率 | active ratio | 超过阈值的通道比例，反映 MLP 稀疏性 |
| 范数 | L2 norm | 向量的「长度」`√(Σx²)`，日志里用来摘要向量大小 |

---

## A.4 分词术语

| 术语 | 英文 | 含义 |
|------|------|------|
| 分词器 | tokenizer | 文本和 token id 之间的桥梁 |
| token | — | 模型处理的最小单位（一个词 / 一个字 / 一块字节） |
| token id | — | token 在词表里的整数编号 |
| BPE | Byte-Pair Encoding | 字节对编码分词，反复合并最高频相邻对 |
| pre-token | — | BPE 前先切的文本片段（如按空格切词） |
| byte-level | — | 字节 → unicode 映射（GPT-2 的解法） |
| merge | — | BPE 的合并规则 |
| vocab | vocabulary | 词表，所有 token 的集合 |
| BOS | Beginning Of Sequence | 序列起始符 |
| EOS | End Of Sequence | 序列结束符（生成时遇到就停） |
| PAD | Padding | 填充符（batch 对齐用） |
| 哈希表 | hash table | 用哈希函数加速「字符串 → id」查找 |
| FNV-1a | — | 一种简单快速的字符串哈希算法 |
| 拉链法 | chaining | 哈希冲突时用链表串联的解法 |
| 哈希冲突 | hash collision | 不同字符串算出同样的哈希值 |
| teacher forcing | 教师强制 | prefill 阶段按标准答案走，不采样 |

---

## A.5 数据格式术语

| 术语 | 英文 | 含义 |
|------|------|------|
| safetensors | — | HuggingFace 的二进制张量格式（安全 + mmap 友好） |
| dtype | data type | 张量数据类型 |
| fp32 | float32 | 32 位标准浮点（符号 1 + 指数 8 + 尾数 23） |
| bf16 | bfloat16 | 16 位浮点（符号 1 + 指数 8 + 尾数 7），fp32 砍低 16 位 |
| fp16 | float16 | 16 位浮点（符号 1 + 指数 5 + 尾数 10），动态范围小 |
| mmap | memory map | 内存映射文件，访问内存等于读文件 |
| little-endian | 小端序 | 低位字节存在低地址 |
| header | — | safetensors 开头的 JSON 元信息表 |
| raw buffer | — | safetensors 头之后的连续二进制张量数据 |
| data_offsets | — | 张量在 raw buffer 中的 `[start, end)` 区间 |
| 量化 | quantization | 用更少位数近似表示权重（如 int4） |
| int4 / int8 | — | 4 位 / 8 位整数表示 |
| scale | 缩放因子 | 量化时把浮点范围压到整数区间的除数 |

---

## A.6 C 语言 / 工程术语

| 术语 | 含义 |
|------|------|
| row-major | 矩阵按行存储（C 默认），`W[i*n+j]` 是第 i 行第 j 列 |
| strict aliasing | 严格别名规则，`float*` 和 `uint32_t*` 指同一块内存是未定义行为 |
| memcpy | 内存拷贝，编译器会优化成零开销（绕过 strict aliasing 的标准做法） |
| 递归下降 | 解析器自上而下递归（如 JSON 解析遇到 `{` 调 parse_object） |
| RFC | 互联网标准文档（如 RFC 8259 = JSON 标准） |
| 反转义 | unescape，把 `\"` 还原成 `"` |
| 代理对 | surrogate pair，UTF-16 用两个 16 位编码表示一个罕见字符 |
| BMP | Basic Multilingual Plane，Unicode 基本平面（码点 0-65535） |
| 拉链法 | 哈希冲突用链表串联 |
| FNV-1a | 一种字符串哈希算法 |
| xorshift | 一种伪随机数算法（用异或和移位） |
| PRNG | Pseudo Random Number Generator，伪随机数生成器 |
| seed | 种子，PRNG 的初始状态 |
| argmax | 取最大值的索引 |
| 轮盘赌 | roulette wheel，按概率分布采样 |
| top-k | 只保留概率最大的 k 个 |
| temperature | 温度，softmax 前对 logits 缩放的参数 |
| softmax | 把分数归一化成概率的函数 |
| ANSI 转义码 | 终端颜色控制序列（`\033[31m` 等） |
| TTY | TeleTYpewriter，终端 |
| isatty | 判断文件描述符是不是终端 |
| stderr | 标准错误流（日志走这里） |
| stdout | 标准输出流（生成文本走这里） |
| va_list | C 可变参数机制（让 `printf("a=%d", a)` 工作） |
| va_start / va_end | 初始化 / 结束 va_list 遍历 |
| vsnprintf | printf 的 va_list 版本 |
| ASan | AddressSanitizer，编译器自带的内存错误检测器 |
| UBSan | UndefinedBehaviorSanitizer，未定义行为检测器 |
| sanitizer | 内存 / 行为检测工具的统称 |

---

## A.7 生态与性能术语

| 术语 | 英文 | 含义 |
|------|------|------|
| 推理 | inference | 用训练好的模型算出结果（只做前向，不更新权重） |
| 前向传播 | forward pass | 从第 1 层算到第 N 层 |
| 反向传播 | backward pass | 训练时从输出算梯度回传，本项目不做 |
| prefill | 预填 | 把 prompt 的 KV Cache 全部填满的阶段 |
| decode | 解码 | 自回归一格一格生成新 token 的阶段 |
| TTFT | time-to-first-token | 生成第一个 token 的延迟（prefill 开销） |
| Continuous Batching | 连续批处理 | 每一步把所有活跃请求拼成一个 batch，GPU 永不空转 |
| PagedAttention | 分页注意力 | 把 KV Cache 切成固定大小的页按需分配（vLLM 创新） |
| Prefix Caching | 前缀缓存 | 缓存相同前缀的 KV Cache，复用算力 |
| RadixAttention | 基数树注意力 | SGLang 用基数树缓存前缀，比哈希表缓存更多组合 |
| FlashAttention | 闪速注意力 | 分块计算 attention，显存从 O(seq²) 降到 O(seq) |
| Speculative Decoding | 投机解码 | 小模型猜 + 大模型验证，一次 forward 出多个 token |
| Tensor Parallelism | 张量并行 | 把大矩阵按行 / 列切到多卡 |
| Pipeline Parallelism | 流水线并行 | 按层切到多卡，每卡负责几层 |
| CPU Offloading | CPU 卸载 | 权重放内存，用到哪层搬到哪层 |
| MoE | Mixture of Experts | 混合专家，每次只用一部分专家权重 |
| SIMD | Single Instruction Multiple Data | 单指令多数据，CPU 并行（AVX / NEON） |
| Tensor Core | — | GPU 上专门做矩阵乘法的硬件单元 |
| cuBLAS | — | NVIDIA 的线性代数库（GPU 版 BLAS） |
| cuDNN | — | NVIDIA 的深度学习算子库 |
| GGUF | — | llama.cpp 的模型格式（含量化权重） |
| ggml | — | llama.cpp 自研的张量计算库 |
| GQA | Grouped Query Attention | 多个 Q 头共享 KV（见 A.2） |
| MLA | Multi-head Latent Attention | DeepSeek 的注意力结构，替代 GQA |
| All-Gather | — | 多卡通信原语，每张卡拿到所有卡的部分结果拼起来 |
| logits | — | 最后投影到词表的原始分数（softmax 之前） |
| reference 实现 | reference implementation | 对照基准（本项目用 PyTorch transformers） |
| top-5 logits | — | 分数最高的 5 个 token，用于验证前向传播 |
| tie_word_embeddings | — | 输出头复用 embedding 权重（Qwen 是 true，没有 lm_head） |

---

## 速查：本项目关键数字

| 项 | 数值 |
|---|---|
| 参数量 | 494,032,768（≈0.49B，叫「0.5B」） |
| safetensors 文件大小 | 988 MB（bf16 存储） |
| fp32 内存占用 | 1.97 GB |
| dim（特征维） | 896 |
| hidden_dim（MLP 中间宽） | 4864 |
| n_layers | 24 |
| n_heads（Q 头） | 14 |
| n_kv_heads（KV 头） | 2 |
| head_dim | 64 |
| vocab_size | 151936 |
| seq_len | 2048 |
| rope_theta | 1,000,000 |
| rms_eps | 1e-6 |
| BOS / EOS id | 151643（`<|endoftext|>`） |
| 速度（CPU fp32） | ~3 tok/s |
| 速度（vLLM A100 并发） | ~8000 tok/s |

---

> 这份术语表是全书（第 1-10 章）的索引。读任何一章遇到不懂的词，先来这里查；
> 查不到的，去对应章节的正文看详解。
