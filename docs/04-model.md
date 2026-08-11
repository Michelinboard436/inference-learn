# 第四章 模型结构（net.h）

> 第 4 章 | 配合源码 `net.h`（106 行） | [上一章：JSON 解析器](./03-json.md) | [下一章：前向传播](./05-forward.md)

---

## 这章讲什么

`net.h` 是整个推理引擎的「骨架」——它定义了三大数据结构：

- **Config**：模型长什么样（所有维度参数）
- **TransformerWeights**：权重有哪些（全部可学习参数）
- **RunState**：计算时需要什么内存（工作缓冲区）

但里面全是术语：残差流、投影、GQA、RoPE、SwiGLU、KV Cache……

本章先快速过一遍三个结构体（4.1～4.3），再用剩下的小节**逐个讲透每个术语**——
每个术语配三样东西：**大白话解释 + 为什么需要 + 代码在哪**。

> 读法建议：先看 4.1～4.3 建立骨架印象，再按需查 4.4 的术语详解。
> 读完这章再回头看 `net.h`，会有「原来如此」的感觉。

---

## 4.1 Config — 静态配置（模型长什么样）

所有维度参数，来自 `config.json`，运行时不变：

```c
typedef struct {
    int dim;          // 896    残差流宽度 (token 向量长度)
    int hidden_dim;   // 4864   MLP 中间宽度
    int n_layers;     // 24     Transformer 层数
    int n_heads;      // 14     Query 头数
    int n_kv_heads;   // 2      KV 头数 (GQA, 比 n_heads 小)
    int head_dim;     // 64     每个头的维度
    int vocab_size;   // 151936 词表大小
    int seq_len;      // 2048   最大序列长度
    float rope_theta; // 1e6    RoPE 基频
    float rms_eps;    // 1e-6   RMSNorm 防除零
} Config;
```

**对照 `config.json` 的字段**（两者一一对应）：

| net.h 字段 | config.json 字段 | 值 |
|-----------|------------------|-----|
| `dim` | `hidden_size` | 896 |
| `hidden_dim` | `intermediate_size` | 4864 |
| `n_layers` | `num_hidden_layers` | 24 |
| `n_heads` | `num_attention_heads` | 14 |
| `n_kv_heads` | `num_key_value_heads` | 2 |
| `head_dim` | （由 `dim/n_heads` 算出） | 64 |
| `vocab_size` | `vocab_size` | 151936 |
| `rope_theta` | `rope_theta` | 1,000,000 |
| `rms_eps` | `rms_norm_eps` | 1e-6 |

> ⭐ **先理解三个数字就能入门**：`dim=896`（词向量多长）、`n_layers=24`（多少层）、
> `vocab_size=151936`（认识多少词）。其他数字大多是从这三个派生的。

---

## 4.2 TransformerWeights — 所有权重（模型学了什么）

权重按层堆叠成连续数组，方便用 `layer * stride + offset` 索引：

```c
typedef struct {
    float *token_embedding_table;  // (151936, 896) 查表: token→向量
                                   // 也是输出头(tie_word_embeddings)
    float *rms_att_weight;         // (24, 896)     attention 前的归一化
    float *rms_ffn_weight;         // (24, 896)     MLP 前的归一化

    float *wq;   // (24, 896, 896)  Query 投影
    float *wk;   // (24, 128, 896)  Key 投影   ← 维度小 (GQA)
    float *wv;   // (24, 128, 896)  Value 投影 ← 维度小 (GQA)
    float *wo;   // (24, 896, 896)  Output 投影

    float *bq;   // (24, 896)  Query 偏置  ← Qwen 独有!
    float *bk;   // (24, 128)  Key 偏置    ← Qwen 独有!
    float *bv;   // (24, 128)  Value 偏置  ← Qwen 独有!
    // 注意: wo 没有 bias (Qwen 设计)

    float *w_gate; // (24, 4864, 896) 门控: 决定哪些通道激活
    float *w_up;   // (24, 4864, 896) 上投: 提供候选值
    float *w_down; // (24, 896, 4864) 下投: 压回残差流宽度

    float *rms_final_weight; // (896,) 最终归一化
} TransformerWeights;
```

**Qwen 特有点（和 Llama 的关键区别）**：

- Q/K/V 投影**有** bias（`bq/bk/bv`），O 投影**没有** bias
- 这是因为 Qwen 训练时在 QKV 上加了 bias，Llama 系列全部无 bias

**单层权重维度速查表**：

| 权重 | 形状 | 元素数 | 方阵? | 说明 |
|------|------|--------|-------|------|
| Q 投影 (wq) | `(896, 896)` | 802,816 | ✅ | 输入输出都是 896 维 |
| K 投影 (wk) | `(128, 896)` | 114,688 | ❌ | GQA 让输出只有 128 维 |
| V 投影 (wv) | `(128, 896)` | 114,688 | ❌ | 同 K |
| O 投影 (wo) | `(896, 896)` | 802,816 | ✅ | 整合 14 个头 |
| MLP gate/up | `(4864, 896)` | 4,358,144 | ❌ | 扩张到 4864 维 |
| MLP down | `(896, 4864)` | 4,358,144 | ❌ | 压回 896 维 |

> ⚠️ **常见误区**：`hidden_size=896` 不代表模型里所有矩阵都是 896×896 的方阵。
> 它只表示「每个 token 的特征向量有 896 个数」。至于矩阵长什么样，取决于这个矩阵做什么。

---

## 4.3 RunState — 工作内存（计算时用什么）

前向传播时的临时缓冲区，每次 forward 复用：

```c
typedef struct {
    float *x;      // (896)     当前层输入
    float *xb;     // (896)     归一化后/attention 内部用
    float *xb2;    // (896)     MLP 内部用 / 投影输出暂存
    float *hb;     // (4864)    MLP gate 结果
    float *hb2;    // (4864)    MLP up 结果
    float *q;      // (896)     当前 token 的 Query
    float *k;      // (128)     当前 token 的 Key
    float *v;      // (128)     当前 token 的 Value
    float *att;    // (14, 2048) 注意力分数
    float *logits; // (151936)  最终输出分数

    /* ====== KV Cache (跨步累积) ====== */
    float *key_cache;   // (24, 2048, 128) 所有历史位置的 Key
    float *value_cache; // (24, 2048, 128) 所有历史位置的 Value
} RunState;
```

**为什么 Weights 和 RunState 分开**：

- **Weights 只读**：模型参数，整个生成过程都不变
- **RunState 读写**：每个 forward 步骤都被覆盖

分开管理更清晰，也方便多线程（一份权重多个线程复用，各自独立的 RunState）。

> 类比厨房：
> - **Weights = 食材**（菜谱 + 原料）：一次性备齐，全程复用
> - **RunState = 厨具**（案板、锅、刀）：做每道菜都用，做完洗洗下道菜接着用

---

## 4.4 术语详解（核心价值）

下面把 `net.h` 里所有术语分组讲透。每组术语都配「大白话 + 为什么 + 代码在哪」。

### 第一组：张量与维度（最基础）

#### dim（维度 / 残差流宽度）

**是什么**：模型内部表示一个 token 时用的向量长度。Qwen2.5-0.5B 中 `dim = 896`。

**大白话**：想象每个 token 是一个 896 格的「档案卡」，每格存一个浮点数。
这 896 个数字合在一起，就是模型对这个词的「理解」。数字越多，理解越精细。

**为什么是这个数**：模型设计者选的。越大 = 表达力越强，但计算量越大。
0.5B 模型用 896，7B 模型通常用 4096。

**在 net.h**：

```c
// Config 结构体, 第 24 行
int dim;          // 残差流宽度 896
```

**在代码里怎么用**：贯穿全场。比如分配当前层的输入向量：

```c
s->x = calloc(dim, sizeof(float));   // 一个 896 格的向量
```

> ⚠️ **易混点**：`dim=896` 是**特征维（列数）**，不是行数。
> 行数由「输入了几个 token」（seq_len）决定，和 896 完全无关。
> 输入 2 个 token → 数据张量形状是 `[2, 896]`（2 行 × 896 列）。

---

#### hidden_dim（MLP 中间宽度 / intermediate_size）

**是什么**：MLP（前馈网络）中间层的维度。Qwen2.5-0.5B 中 `hidden_dim = 4864`。

**大白话**：残差流是 896 维的「主干道」。到了 MLP，先把它**扩张**到 4864 维
（做更丰富的计算），再**压缩**回 896 维。像手的「伸开→抓拢」。

```
896 维 ──扩张──→ 4864 维 ──激活──→ 4864 维 ──压缩──→ 896 维
         W_gate            SwiGLU            W_down
```

**为什么是 4864**：约为 896 的 5.4 倍。SwiGLU 因为多一个门控矩阵（见第五组），
扩张比传统 MLP 稍小。常见倍率是 4 左右。

**在 net.h**：

```c
// 第 25 行
int hidden_dim;   // MLP 中间宽度 4864
```

---

#### head_dim（每个注意力头的维度）

**是什么**：注意力被切成多个「头」后，每个头的向量长度。`head_dim = 64`。

**大白话**：896 维的向量不方便直接做注意力（太粗）。把它切成 14 份，每份 64 维，
每个「头」独立关注不同方面的信息——一个头看语法，一个头看语义……

```
896 维 Q 向量
├── head 0 (64维): 这 64 个数做一次注意力
├── head 1 (64维): 另外 64 个数做一次注意力
├── ...
└── head 13 (64维)
共 14 个头 × 64 维 = 896 维
```

**计算**：`head_dim = dim / n_heads = 896 / 14 = 64`

**在 net.h**：

```c
// 第 29 行
int head_dim;     // 每个头的维度 64 = dim / n_heads
```

---

#### n_layers（Transformer 层数）

**是什么**：Transformer 堆叠了多少层。`n_layers = 24`。

**大白话**：一层 Transformer 做「注意 + 前馈」一次。24 层就是做 24 次。
层数越多，模型能学到的关系越深、越抽象。

```
token → [Layer 0] → [Layer 1] → ... → [Layer 23] → logits
         浅层: 学词性、语法     深层: 学逻辑、推理
```

**类比**：像工厂流水线的工位数。每个工位（层）都加工一下半成品，24 道工序后产出。

**在 net.h**：

```c
// 第 26 行
int n_layers;     // Transformer 层数 24
```

---

#### vocab_size（词表大小）

**是什么**：模型认识多少种 token。`vocab_size = 151936`。

**大白话**：模型的「字典」有 151,936 个条目。每个 token 是字典里的一个词或字片段。
embedding 表就是这本字典的「含义页」——每个 token 对应 896 个数字。

**为什么这么大**：Qwen 支持多语言（中英日韩……）、代码、特殊符号，所以词表大。
GPT-2 只有 50,257。

**在 net.h**：

```c
// 第 30 行
int vocab_size;   // 151936
// 和 embedding 矩阵的大小直接相关:
float *token_embedding_table;  // (vocab_size, dim) = (151936, 896)
```

---

### 第二组：Embedding（文字怎么变数字）

#### token_embedding（词嵌入 / Token Embedding）

**是什么**：一个 `(151936, 896)` 的查找表。给定 token id，查出一个 896 维向量。

**大白话**：模型的「字典」。输入是 token 编号，查表得到这个词的「含义向量」。

```
token id = 9707 ("Hello")
         ↓ 查 token_embedding_table[9707]
得到 [0.03, -0.2, 0.5, ..., 0.01]  (896 个数)
```

**代码**：

```c
// net.c: forward() 中
float *embed_row = w->token_embedding_table + (size_t)token * dim;
memcpy(s->x, embed_row, dim * sizeof(float));
```

---

#### tie_word_embeddings（权重共享）

**是什么**：输入端（embedding）和输出端（分类头）**用同一个矩阵**。

**大白话**：入口「token → 向量」和出口「向量 → token 概率」用同一本字典。
入口查「这个词什么意思」，出口问「哪个词和这个意思最接近」——用同一个表更省。

```
入口:  token id 9707 ──查表──→ 896维向量 (理解输入)
出口:  896维向量 ──和表做点积──→ 151936个分数 (选最匹配的token)
                    ↑
              同一个 token_embedding_table
```

**为什么**：省 151936 × 896 × 4 字节 ≈ 544 MB 内存。小模型常用。

**在 net.h**：

```c
// 第 19 行注释 + 第 45 行
// tie_word_embeddings=true → lm_head 复用 token_embedding，不单独存储
float *token_embedding_table;  // 既是 embedding 也是 lm_head
```

---

#### logits（未归一化的分数）

**是什么**：最后一层输出，对每个 token 的「原始分数」。长度 = vocab_size = 151936。

**大白话**：模型对「下一个词应该是谁」打的分。分数越高越可能。
但 logits 不是概率（可以是负数、可以总和不为 1），需要 softmax 才变成概率。

```
logits = [-2.3, 0.5, 15.8, -1.1, ..., 8.2]   ← 151936 个分数
                                       ↑
                              id=2 ('B') 分数最高
```

**代码**：

```c
// net.c 最后一步
for (int v = 0; v < config->vocab_size; v++) {
    float *embrow = w->token_embedding_table + v * dim;
    for (int i = 0; i < dim; i++)
        s->logits[v] += s->x[i] * embrow[i];   // 当前向量 × 词表第v行
}
```

---

### 第三组：注意力机制（Attention）

#### Q / K / V（Query / Key / Value）

**是什么**：注意力的三个核心向量，通过对输入做线性变换得到。

**大白话**：用「图书馆检索」类比：

| | 图书馆类比 | 作用 |
|---|---|---|
| **Query (Q)** | 你的搜索词「我想找秋天的诗」 | 当前 token 在「问」什么 |
| **Key (K)** | 每本书的标签/书名 | 每个 token「是」什么 |
| **Value (V)** | 书的实际内容 | 每个 token 的实际信息 |

Q 和 K 点积 = 相关程度。越相关，取越多那个 token 的 V。

**代码**：每个 token 的输入 x 经过三个不同的权重矩阵变换：

```c
Q = x @ Wq + bq    // "我想找什么"
K = x @ Wk + bk    // "我有什么"
V = x @ Wv + bv    // "我的内容"
```

---

#### 投影 / Projection（Wq, Wk, Wv, Wo）

**是什么**：把输入向量乘以一个权重矩阵，转换到另一个空间。

**大白话**：「投影」就是矩阵乘法。`Wq` 把 896 维的输入「翻译」成 Query 用的格式。
想象调整望远镜的焦距——同一个画面，换不同镜头看到不同信息。

```
x (896维, 通用表示)
 ├── @ Wq → Q (896维, "提问视角")
 ├── @ Wk → K (128维, "标签视角")
 └── @ Wv → V (128维, "内容视角")
```

**注意**：Wq 输出 896 维（=14头×64），Wk/Wv 只输出 128 维（=2头×64）——这就是 GQA。

**在 net.h**：

```c
// 第 51-54 行
float *wq;   // Query 投影权重, (24, 896, 896)
float *wk;   // Key 投影权重,   (24, 128, 896)  ← 比 wq 小很多
float *wv;   // Value 投影权重, (24, 128, 896)
float *wo;   // Output 投影权重,(24, 896, 896)  把注意力输出合回 896 维
```

---

#### bias（偏置）

**是什么**：矩阵乘法后额外加的一个向量。`y = Wx + b` 里的 `b`。

**大白话**：投影矩阵做的是「旋转+缩放」，bias 做的是「平移」。
好比调音响：W 调音色，b 调音量基准。

**Qwen 的特殊性**：Q/K/V 投影**有** bias，O 投影**没有**。这和 Llama 不同。

| | Llama | Qwen |
|---|---|---|
| Q/K/V 的 bias | ❌ 无 | ✅ 有 |
| O 的 bias | ❌ 无 | ❌ 无 |

**在 net.h**：

```c
// 第 56-59 行
float *bq;   // Query 偏置,  (24, 896)   ← Qwen 独有
float *bk;   // Key 偏置,    (24, 128)   ← Qwen 独有
float *bv;   // Value 偏置,  (24, 128)   ← Qwen 独有
// 注意: wo 没有 bias
```

> **为什么 bias 维度不同**：bias 维度 = 输出维度。Q 输出 896 维所以 bias 是 896，
> K/V 输出 128 维所以 bias 是 128。

---

#### GQA（Grouped Query Attention，分组查询注意力）

**是什么**：14 个 Query 头，但只有 2 个 Key/Value 头。每 7 个 Q 头共享 1 组 KV。

**大白话**：标准注意力是「14 个提问者各配 14 个资料柜」。GQA 是「14 个提问者共享 2 个资料柜」。

```
标准 MHA:  14 个 Q 头, 14 套 KV  → KV cache 巨大, 但质量最好
MQA:       14 个 Q 头,  1 套 KV  → KV cache 最小, 但质量降
GQA:       14 个 Q 头,  2 套 KV  → 折中 ★ Qwen2.5 用这个

Qwen2.5-0.5B: head 0-6 共享 KV头0, head 7-13 共享 KV头1
```

**为什么**：KV Cache 内存从 14 份降到 2 份（**省 7 倍**），质量损失很小。

**在 net.h**：

```c
// 第 27-28 行
int n_heads;      // Q 头数 14
int n_kv_heads;   // KV 头数 2 (GQA)
```

**在代码里**：

```c
int group = n_heads / n_kv_heads;  // = 7
for (int h = 0; h < n_heads; h++) {
    int kvh = h / group;           // head 0-6 → kvh=0, head 7-13 → kvh=1
    // 用 KV 头 kvh 的数据算注意力
}
```

---

### 第四组：归一化（Normalization）

#### RMSNorm（Root Mean Square Normalization，均方根归一化）

**是什么**：把向量除以它的均方根（RMS），再逐元素乘以一个可学习权重。

**大白话**：控制向量的「音量」不要太大或太小。防止经过多层后数值爆炸或消失。

```
RMSNorm(x):
  1. 算 RMS = sqrt(mean(x²))
  2. y = x / RMS           ← 归一化到合理大小
  3. y = y * weight        ← 每维乘一个可学习的缩放系数
```

**对比 LayerNorm**：RMSNorm 省掉了「减均值」这一步，计算更少，效果差不多。

| | LayerNorm | RMSNorm |
|---|---|---|
| 减均值 | ✅ | ❌ |
| 除标准差 | ✅ | ❌ |
| 除 RMS | ❌ | ✅ |
| 现代 LLM | 早期 | **主流**（Llama/Qwen）|

**在 net.h**：每层有**两个** RMSNorm——一个在 attention 前，一个在 MLP 前。

```c
// 第 48-49 行
float *rms_att_weight;   // attention 前的归一化权重 (24, 896)
float *rms_ffn_weight;   // MLP 前的归一化权重     (24, 896)
// 第 66 行
float *rms_final_weight; // 全部层结束后的最终归一化 (896,)
```

> 为什么每层要两次？一层里有 Attention 和 MLP 两个子模块，各自都做矩阵乘法，
> 所以每次矩阵乘法前都要归一化。详见第 5 章 5.2 节。

---

#### rms_eps（归一化的小常数）

**是什么**：RMSNorm 分母里的 ε，防止除零。`rms_eps = 1e-6`。

**大白话**：万一向量全是 0，RMS = 0，除以 0 会得到无穷大。加个极小的数兜底。

```c
float inv = 1.0f / sqrtf(mean(x²) + 1e-6);   // ← 这里的 1e-6
```

---

### 第五组：前馈网络（MLP / FFN）

#### MLP / FFN（Multi-Layer Perceptron / Feed-Forward Network）

**是什么**：每层 Transformer 里，注意力之后的一个「两层全连接网络」。

**大白话**：注意力负责「token 之间交换信息」，MLP 负责「每个 token 自己加工信息」。
注意力是「开会讨论」，MLP 是「回工位独立思考」。

**结构**：扩张 → 激活 → 压缩

```
896维 ──→ 4864维 ──→ 激活(非线性) ──→ 896维
         扩张        SwiGLU          压缩
```

> MLP 占了每层 88% 的参数。如果想压缩模型，MLP 是首要目标（也是 MoE 改革它的原因）。

---

#### SwiGLU（Swish-Gated Linear Unit，门控线性单元）

**是什么**：Qwen 用的激活函数。`output = silu(gate × x) * (up × x)`。

**大白话**：传统激活是「一个输入过一个函数」。SwiGLU 多了一条「门控」支路：
gate 决定**哪些通道该激活**，up 提供**候选值**，两者相乘。

```
         ┌── @ W_gate → gate ── silu() ──┐
x (896维) ┤                                × (逐元素相乘) → 4864维 → @ W_down → 896维
         └── @ W_up   → up   ────────────┘
```

**为什么比 ReLU 好**：

- ReLU 是硬门（负数直接归零，正数原样输出）
- SwiGLU 是软门（用 sigmoid 平滑控制，且门控值是动态学习的）

**三个权重矩阵**：

```c
// net.h 第 62-64 行
float *w_gate; // 门控: (24, 4864, 896) 决定"开多少"
float *w_up;   // 上投: (24, 4864, 896) 提供"候选值"
float *w_down; // 下投: (24, 896, 4864) 压回 896 维
```

---

#### silu（SiLU / Swish 激活函数）

**是什么**：`silu(x) = x × sigmoid(x) = x / (1 + e^(-x))`。

**大白话**：比 ReLU 平滑。负数不是直接归零而是缓慢趋近零，正数近似恒等。

```
ReLU(x):  负数→0, 正数→原值     (有棱角, 梯度可能断裂)
silu(x):  负数→平滑趋近0        (平滑, 处处可导)
          正数→近似 x
```

---

### 第六组：位置编码（RoPE）

#### RoPE（Rotary Position Embedding，旋转位置编码）

**是什么**：通过「旋转」向量的某些维度来编码位置信息。`rope_theta = 1,000,000`。

**大白话**：注意力本身不知道 token 的顺序（"猫咬人"和"人咬猫"对它一样）。
RoPE 通过给每个位置旋转不同角度，让注意力能感知「相隔多远」。

```
位置 0 的向量: 不转
位置 1 的向量: 转一点
位置 2 的向量: 转两倍
...

两个 token 的注意力分数 = 它们旋转角度的差 = 相对位置
```

**rope_theta 的作用**：基频。越大，能编码的位置越远（支持更长上下文）。

- GPT-2: theta=10000（上下文 1024）
- **Qwen2.5: theta=1000000**（上下文 32768）

**在 net.h**：

```c
// 第 32 行
float rope_theta; // 1,000,000
```

**只作用于 Q 和 K**，不作用于 V——位置信息只在计算相关性（Q·K）时需要。

---

### 第七组：KV Cache（生成加速）

#### KV Cache（键值缓存）

**是什么**：把已计算过的每个 token 的 Key/Value 存起来，下次直接复用。

**大白话**：生成第 N 个词时，前 N-1 个词的 K/V 已经算过了，不用重算。
就像做菜——切好的菜放冰箱，下次直接用，不用重新切。

```
没有 KV Cache: 生成第 5 个词, 要重新算前 4 个词的 K/V  → O(N²)
有 KV Cache:   生成第 5 个词, 只算第 5 个词的 K/V     → O(N)
               前 4 个的 K/V 从缓存读
```

**在 net.h**：

```c
// 第 85-86 行
float *key_cache;   // (24层, 2048位置, 128维) 存所有历史 K
float *value_cache; // (24层, 2048位置, 128维) 存所有历史 V
```

**内存**：2 × 24 × 2048 × 128 × 4 字节 = 50 MB。
如果不用 GQA（用 14 头），会是 353 MB——这就是 GQA 的价值。

---

#### seq_len（序列长度）

**是什么**：我们支持的最大 token 数。`seq_len = 2048`（代码里设的，模型本身支持 32768）。

**大白话**：对话能塞多少个 token。KV Cache 的大小直接由它决定。
设小一点省内存，设大一点能处理更长的文本。

---

### 第八组：其他概念

#### 残差连接（Residual Connection）

**是什么**：每层的输出 = 输入 + 该层的计算结果。`x = x + layer(x)`。

**大白话**：每层不直接替换输入，而是在输入基础上「叠加」一点修正。
像批改作业——不是重写，是在原文上加批注。

```c
// net.c 里每层有两处残差
s->x[i] += attention_out;   // 注意力结果加回残差
s->x[i] += mlp_out;         // MLP 结果加回残差
```

**为什么需要**：没有残差，深层网络（24层）会梯度消失，训练不动。
残差让信息能直接「跳过」某些层，梯度也能直接回流。

---

#### 前向传播（Forward Pass / Forward）

**是什么**：给定输入，从第一层算到最后一层，得到输出。

**大白话**：模型「思考」一次的过程。输入一个 token，一路算到 logits。

```c
// net.h 第 103 行
void forward(Config *config, TransformerWeights *w, RunState *s,
             int token, int pos);
```

**对比**：训练时还有「反向传播」（backward），算梯度更新权重。
**推理引擎只需要前向**——所以比训练简单很多。详见第 5 章。

---

#### bf16 → fp32（数据类型转换）

**是什么**：权重用 bfloat16 存储（省空间），计算用 float32（精度高）。

**大白话**：模型存盘用「压缩格式」（bf16，每个数 2 字节），
计算时解压成「原始格式」（fp32，每个数 4 字节）。

```
bf16: 16位 = 1符号 + 8指数 + 7尾数    (省空间, 精度够)
fp32: 32位 = 1符号 + 8指数 + 23尾数   (精度高, 占双倍空间)

转换: bf16 左移 16 位 = fp32 的高 16 位 (它们共享同样的指数格式)
```

详见第 2 章（safetensors 加载）的 2.3 节。

---

## 4.5 一张图串起每层各参数的关系

```
输入 x (896维)
  │
  ├─ 归一化 rms_att_weight (896) ──→ 调好数值大小 (防爆炸)
  │        ↓
  │  Q/K/V 投影 wq/wk/wv + bq/bk/bv ──→ 算出三种角色 (bias 帮平移)
  │        ↓
  │     RoPE 旋转 (只作用 Q/K) ──→ 编码位置
  │        ↓
  │     存 K/V 进 KV Cache
  │        ↓
  │     Attention 计算 (14个Q头 注意 2个KV头, GQA)
  │        ↓
  │  O 投影 wo (无 bias) ──→ 整合 14 个头
  │        ↓
  ├─ 残差连接 (x += attention_out)
  │        ↓
  ├─ 归一化 rms_ffn_weight (896) ──→ 再调一次数值
  │        ↓
  │  MLP w_gate + w_up ──→ 扩张到 4864 维
  │     silu(gate) × up  (SwiGLU 门控)
  │     w_down ──→ 压回 896 维
  │        ↓
  └─ 残差连接 (x += mlp_out)
       ↓
     输出给下一层 (重复 24 次)
       ↓
     最终归一化 rms_final_weight + 投影回词表 → logits
```

每个箭头都对应 `net.c` 里的一段计算。具体每个算子怎么算，见第 5 章。

---

## 4.6 学习建议

1. **先理解三个数字**：`dim=896`（词向量多长）、`n_layers=24`（多少层）、
   `vocab_size=151936`（认识多少词）。其他数字都是从这三个派生的。

2. **理清两条线**：
   - **数据线**（RunState）：一个 token 怎么在网络里流动
   - **权重线**（TransformerWeights）：网络存了哪些可学习参数

3. **动手验证**：用 `./run fwd model.safetensors 198 0 -v` 跑一次，
   对照日志里的中间值，看每个术语对应哪一步。

4. **和 config.json 对照**：`net.h` 里的 Config 结构体就是 config.json 的 C 语言映射，
   两者一一对应。

---

> 下一章我们进入引擎的核心——`net.c` 的前向传播。所有算子（RMSNorm、matmul、
> RoPE、Attention、SwiGLU）都会在那里逐行拆解，配真实数字走完整流程。
> [继续阅读第 5 章 →](./05-forward.md)
