> 🌐 [中文版](../04-model) | English

# Chapter 4 · Model Architecture (net.h)

> Chapter 4 | Companion source: `net.h` (106 lines) | [Previous: JSON Parser](./03-json.md) | [Next: Forward Pass](./05-forward.md)

---

## What This Chapter Covers

`net.h` is the "skeleton" of the entire inference engine — it defines the three core data structures:

- **Config**: what the model looks like (all the dimension parameters)
- **TransformerWeights**: what weights there are (all the learnable parameters)
- **RunState**: what memory is needed for computation (the working buffers)

But it's full of jargon: residual stream, projection, GQA, RoPE, SwiGLU, KV Cache...

This chapter first walks quickly through the three structs (4.1–4.3), then uses the remaining sections to **explain every term thoroughly** — each term comes with three things: **plain-English explanation + why it's needed + where it lives in the code**.

> How to read: skim 4.1–4.3 first to get a skeletal impression, then look up individual terms in 4.4 as needed.
> After finishing this chapter, come back to `net.h` and you'll have a "so that's how it is" moment.

---

## Config — Static Configuration (What the Model Looks Like)

All the dimension parameters, sourced from `config.json`, immutable at runtime:

```c
typedef struct {
    int dim;          // 896    residual stream width (token vector length)
    int hidden_dim;   // 4864   MLP intermediate width
    int n_layers;     // 24     number of Transformer layers
    int n_heads;      // 14     number of Query heads
    int n_kv_heads;   // 2      number of KV heads (GQA, smaller than n_heads)
    int head_dim;     // 64     dimension per head
    int vocab_size;   // 151936 vocabulary size
    int seq_len;      // 2048   maximum sequence length
    float rope_theta; // 1e6    RoPE base frequency
    float rms_eps;    // 1e-6   RMSNorm epsilon to prevent division by zero
} Config;
```

**Mapped onto `config.json` fields** (one-to-one correspondence):

| net.h field | config.json field | Value |
|-----------|------------------|-----|
| `dim` | `hidden_size` | 896 |
| `hidden_dim` | `intermediate_size` | 4864 |
| `n_layers` | `num_hidden_layers` | 24 |
| `n_heads` | `num_attention_heads` | 14 |
| `n_kv_heads` | `num_key_value_heads` | 2 |
| `head_dim` | (computed from `dim/n_heads`) | 64 |
| `vocab_size` | `vocab_size` | 151936 |
| `rope_theta` | `rope_theta` | 1,000,000 |
| `rms_eps` | `rms_norm_eps` | 1e-6 |

> ⭐ **Three numbers are enough to get started**: `dim=896` (how long a word vector is), `n_layers=24` (how many layers),
> `vocab_size=151936` (how many words the model knows). Most of the other numbers are derived from these three.

---

## TransformerWeights — All the Weights (What the Model Has Learned)

The weights are stacked into contiguous arrays by layer, so they can be indexed with `layer * stride + offset`:

```c
typedef struct {
    float *token_embedding_table;  // (151936, 896) lookup: token → vector
                                   // also the output head (tie_word_embeddings)
    float *rms_att_weight;         // (24, 896)     normalization before attention
    float *rms_ffn_weight;         // (24, 896)     normalization before MLP

    float *wq;   // (24, 896, 896)  Query projection
    float *wk;   // (24, 128, 896)  Key projection   ← small dim (GQA)
    float *wv;   // (24, 128, 896)  Value projection ← small dim (GQA)
    float *wo;   // (24, 896, 896)  Output projection

    float *bq;   // (24, 896)  Query bias  ← Qwen-only!
    float *bk;   // (24, 128)  Key bias    ← Qwen-only!
    float *bv;   // (24, 128)  Value bias  ← Qwen-only!
    // note: wo has no bias (Qwen design)

    float *w_gate; // (24, 4864, 896) gate: decides which channels to activate
    float *w_up;   // (24, 4864, 896) up-projection: provides candidate values
    float *w_down; // (24, 896, 4864) down-projection: compresses back to residual width

    float *rms_final_weight; // (896,) final normalization
} TransformerWeights;
```

**Qwen-specific (the key difference from Llama)**:

- Q/K/V projections **have** bias (`bq/bk/bv`); the O projection does **not** have bias
- This is because Qwen added bias on QKV during training; the Llama family has no bias anywhere

**Quick-reference table for a single layer's weight dimensions**:

| Weight | Shape | Element count | Square? | Notes |
|------|------|--------|-------|------|
| Q projection (wq) | `(896, 896)` | 802,816 | ✅ | input and output both 896-dim |
| K projection (wk) | `(128, 896)` | 114,688 | ❌ | GQA makes the output only 128-dim |
| V projection (wv) | `(128, 896)` | 114,688 | ❌ | same as K |
| O projection (wo) | `(896, 896)` | 802,816 | ✅ | merges the 14 heads |
| MLP gate/up | `(4864, 896)` | 4,358,144 | ❌ | expands to 4864-dim |
| MLP down | `(896, 4864)` | 4,358,144 | ❌ | compresses back to 896-dim |

> ⚠️ **Common misconception**: `hidden_size=896` does not mean every matrix in the model is an 896×896 square.
> It only means "each token's feature vector has 896 numbers." What a matrix actually looks like depends on what that matrix does.

---

## RunState — Working Memory (What's Used During Computation)

Temporary buffers during the forward pass, reused on every forward step:

```c
typedef struct {
    float *x;      // (896)     current layer's input
    float *xb;     // (896)     after normalization / used inside attention
    float *xb2;    // (896)     used inside MLP / scratch for projection output
    float *hb;     // (4864)    MLP gate result
    float *hb2;    // (4864)    MLP up result
    float *q;      // (896)     the current token's Query
    float *k;      // (128)     the current token's Key
    float *v;      // (128)     the current token's Value
    float *att;    // (14, 2048) attention scores
    float *logits; // (151936)  final output scores

    /* ====== KV Cache (accumulated across steps) ====== */
    float *key_cache;   // (24, 2048, 128) Keys for all historical positions
    float *value_cache; // (24, 2048, 128) Values for all historical positions
} RunState;
```

**Why Weights and RunState are kept separate**:

- **Weights are read-only**: model parameters, never change throughout generation
- **RunState is read-write**: overwritten on every forward step

Keeping them separate is cleaner and also friendlier to multithreading (one copy of weights shared by many threads, each with its own independent RunState).

> Kitchen analogy:
> - **Weights = ingredients** (recipe + raw materials): prepped once, reused the whole time
> - **RunState = utensils** (cutting board, pot, knife): used for every dish, washed between dishes and reused

---

## Glossary in Depth (the core value)

Below, every term in `net.h` is grouped and explained thoroughly. Each group comes with "plain English + why + where in the code."

### Group 1: Tensors and Dimensions (the most basic)

#### dim (dimension / residual stream width)

**What it is**: the vector length the model uses internally to represent a token. In Qwen2.5-0.5B, `dim = 896`.

**Plain English**: imagine each token is an 896-cell "record card," with each cell storing one float.
Together, these 896 numbers are the model's "understanding" of the word. The more numbers, the finer the understanding.

**Why this number**: chosen by the model designers. Bigger = more expressive power, but more compute.
A 0.5B model uses 896; a 7B model typically uses 4096.

**In net.h**:

```c
// the Config struct, line 24
int dim;          // residual stream width 896
```

**How it's used in the code**: it's everywhere. For example, allocating the current layer's input vector:

```c
s->x = calloc(dim, sizeof(float));   // an 896-cell vector
```

> ⚠️ **Easily confused**: `dim=896` is the **feature dimension (column count)**, not the row count.
> The row count is decided by "how many tokens were input" (seq_len), unrelated to 896.
> Input 2 tokens → the data tensor shape is `[2, 896]` (2 rows × 896 columns).

---

#### hidden_dim (MLP intermediate width / intermediate_size)

**What it is**: the dimension of the MLP (feed-forward network) intermediate layer. In Qwen2.5-0.5B, `hidden_dim = 4864`.

**Plain English**: the residual stream is an 896-dim "main road." Inside the MLP, it first **expands** to 4864-dim
(to do richer computation), then is **compressed** back to 896-dim. Like a hand "opening → closing."

```
896-dim ──expand──→ 4864-dim ──activate──→ 4864-dim ──compress──→ 896-dim
           W_gate              SwiGLU                W_down
```

**Why 4864**: about 5.4× of 896. Because SwiGLU has an extra gate matrix (see Group 5), the expansion ratio is slightly smaller than a traditional MLP. A common ratio is around 4.

**In net.h**:

```c
// line 25
int hidden_dim;   // MLP intermediate width 4864
```

---

#### head_dim (per-attention-head dimension)

**What it is**: after attention is sliced into multiple "heads," the vector length of each head. `head_dim = 64`.

**Plain English**: doing attention directly on an 896-dim vector is inconvenient (too coarse). Slice it into 14 parts of 64-dim each, and each "head" independently focuses on a different aspect of the information — one head looks at syntax, another at semantics...

```
896-dim Q vector
├── head 0 (64-dim): these 64 numbers do one attention
├── head 1 (64-dim): another 64 numbers do one attention
├── ...
└── head 13 (64-dim)
14 heads total × 64-dim = 896-dim
```

**Computation**: `head_dim = dim / n_heads = 896 / 14 = 64`

**In net.h**:

```c
// line 29
int head_dim;     // per-head dimension 64 = dim / n_heads
```

---

#### n_layers (number of Transformer layers)

**What it is**: how many Transformer layers are stacked. `n_layers = 24`.

**Plain English**: one Transformer layer does "attention + feed-forward" once. 24 layers means doing it 24 times.
More layers means the model can learn deeper, more abstract relationships.

```
token → [Layer 0] → [Layer 1] → ... → [Layer 23] → logits
        shallow: parts of speech, syntax    deep: logic, reasoning
```

**Analogy**: like the number of workstations on a factory assembly line. Each station (layer) processes the semi-finished product a bit; after 24 steps, the product is done.

**In net.h**:

```c
// line 26
int n_layers;     // number of Transformer layers 24
```

---

#### vocab_size (vocabulary size)

**What it is**: how many kinds of tokens the model knows. `vocab_size = 151936`.

**Plain English**: the model's "dictionary" has 151,936 entries. Each token is a word or character fragment in the dictionary.
The embedding table is the dictionary's "meaning page" — each token maps to 896 numbers.

**Why so large**: Qwen supports many languages (Chinese, English, Japanese, Korean...), code, and special symbols, so the vocabulary is big.
GPT-2 has only 50,257.

**In net.h**:

```c
// line 30
int vocab_size;   // 151936
// directly tied to the size of the embedding matrix:
float *token_embedding_table;  // (vocab_size, dim) = (151936, 896)
```

---

### Group 2: Embedding (How Text Becomes Numbers)

#### token_embedding (Token Embedding)

**What it is**: a `(151936, 896)` lookup table. Given a token id, it returns a 896-dim vector.

**Plain English**: the model's "dictionary." The input is a token number; a lookup returns the "meaning vector" of that word.

```
token id = 9707 ("Hello")
         ↓ look up token_embedding_table[9707]
get [0.03, -0.2, 0.5, ..., 0.01]  (896 numbers)
```

**Code**:

```c
// in net.c: forward()
float *embed_row = w->token_embedding_table + (size_t)token * dim;
memcpy(s->x, embed_row, dim * sizeof(float));
```

---

#### tie_word_embeddings (weight tying)

**What it is**: the input side (embedding) and the output side (classification head) **use the same matrix**.

**Plain English**: the entrance "token → vector" and the exit "vector → token probability" use the same dictionary.
The entrance looks up "what does this word mean," the exit asks "which word is closest to this meaning" — sharing one table saves memory.

```
entrance:  token id 9707 ──lookup──→ 896-dim vector (understand input)
exit:      896-dim vector ──dot product with table──→ 151936 scores (pick best-matching token)
                    ↑
              the same token_embedding_table
```

**Why**: saves 151936 × 896 × 4 bytes ≈ 544 MB of memory. Common in small models.

**In net.h**:

```c
// line 19 comment + line 45
// tie_word_embeddings=true → lm_head reuses token_embedding, not stored separately
float *token_embedding_table;  // both embedding and lm_head
```

---

#### logits (unnormalized scores)

**What it is**: the output of the last layer, the "raw score" for each token. Length = vocab_size = 151936.

**Plain English**: the model's score for "what the next word should be." The higher the score, the more likely.
But logits are not probabilities (they can be negative, and need not sum to 1) — they need a softmax to become probabilities.

```
logits = [-2.3, 0.5, 15.8, -1.1, ..., 8.2]   ← 151936 scores
                                       ↑
                              id=2 ('B') has the highest score
```

**Code**:

```c
// the last step in net.c
for (int v = 0; v < config->vocab_size; v++) {
    float *embrow = w->token_embedding_table + v * dim;
    for (int i = 0; i < dim; i++)
        s->logits[v] += s->x[i] * embrow[i];   // current vector × row v of the vocab table
}
```

---

### Group 3: Attention

#### Q / K / V (Query / Key / Value)

**What it is**: the three core vectors of attention, obtained by a linear transformation of the input.

**Plain English**: using a "library search" analogy:

| | Library analogy | Role |
|---|---|---|
| **Query (Q)** | your search query "I want autumn poems" | what the current token is "asking" |
| **Key (K)** | each book's tag / title | what each token "is" |
| **Value (V)** | the book's actual content | each token's actual information |

The dot product of Q and K = degree of relevance. The more relevant, the more of that token's V gets pulled in.

**Code**: each token's input x is transformed through three different weight matrices:

```c
Q = x @ Wq + bq    // "what I'm looking for"
K = x @ Wk + bk    // "what I have"
V = x @ Wv + bv    // "my content"
```

---

#### Projection (Wq, Wk, Wv, Wo)

**What it is**: multiplying the input vector by a weight matrix to transform it into another space.

**Plain English**: "projection" is just matrix multiplication. `Wq` "translates" the 896-dim input into the format used for Query.
Imagine adjusting a telescope's focus — the same scene, with different lenses, reveals different information.

```
x (896-dim, general representation)
 ├── @ Wq → Q (896-dim, "asking perspective")
 ├── @ Wk → K (128-dim, "tag perspective")
 └── @ Wv → V (128-dim, "content perspective")
```

**Note**: Wq outputs 896-dim (= 14 heads × 64), Wk/Wv only output 128-dim (= 2 heads × 64) — this is GQA.

**In net.h**:

```c
// lines 51-54
float *wq;   // Query projection weight, (24, 896, 896)
float *wk;   // Key projection weight,   (24, 128, 896)  ← much smaller than wq
float *wv;   // Value projection weight, (24, 128, 896)
float *wo;   // Output projection weight,(24, 896, 896)  merges attention output back to 896-dim
```

---

#### bias

**What it is**: an extra vector added after the matrix multiplication. The `b` in `y = Wx + b`.

**Plain English**: the projection matrix does "rotate + scale"; bias does "translate."
Like adjusting audio gear: W tunes the tone, b tunes the baseline volume.

**Qwen's particularity**: Q/K/V projections **have** bias; the O projection does **not**. This differs from Llama.

| | Llama | Qwen |
|---|---|---|
| Q/K/V bias | ❌ none | ✅ present |
| O bias | ❌ none | ❌ none |

**In net.h**:

```c
// lines 56-59
float *bq;   // Query bias,  (24, 896)   ← Qwen-only
float *bk;   // Key bias,    (24, 128)   ← Qwen-only
float *bv;   // Value bias,  (24, 128)   ← Qwen-only
// note: wo has no bias
```

> **Why the bias dimensions differ**: the bias dimension = the output dimension. Q outputs 896-dim so its bias is 896;
> K/V output 128-dim so their bias is 128.

---

#### GQA (Grouped Query Attention)

**What it is**: 14 Query heads but only 2 Key/Value heads. Every 7 Q heads share 1 group of KV.

**Plain English**: standard attention is "14 questioners, each with their own cabinet of materials." GQA is "14 questioners sharing 2 cabinets of materials."

```
Standard MHA:  14 Q heads, 14 KV sets  → huge KV cache, best quality
MQA:           14 Q heads,  1 KV set   → smallest KV cache, quality drops
GQA:           14 Q heads,  2 KV sets  → a compromise ★ Qwen2.5 uses this

Qwen2.5-0.5B: heads 0-6 share KV head 0, heads 7-13 share KV head 1
```

**Why**: KV Cache memory drops from 14 portions to 2 (**7× savings**), with very little quality loss.

**In net.h**:

```c
// lines 27-28
int n_heads;      // number of Q heads 14
int n_kv_heads;   // number of KV heads 2 (GQA)
```

**In the code**:

```c
int group = n_heads / n_kv_heads;  // = 7
for (int h = 0; h < n_heads; h++) {
    int kvh = h / group;           // heads 0-6 → kvh=0, heads 7-13 → kvh=1
    // compute attention using KV head kvh's data
}
```

---

### Group 4: Normalization

#### RMSNorm (Root Mean Square Normalization)

**What it is**: divides the vector by its root mean square (RMS), then multiplies element-wise by a learnable weight.

**Plain English**: controls the vector's "volume" so it's neither too big nor too small. Prevents values from exploding or vanishing across many layers.

```
RMSNorm(x):
  1. compute RMS = sqrt(mean(x²))
  2. y = x / RMS           ← normalize to a reasonable size
  3. y = y * weight        ← multiply each dim by a learnable scaling coefficient
```

**Compared with LayerNorm**: RMSNorm drops the "subtract mean" step — fewer computations, similar effect.

| | LayerNorm | RMSNorm |
|---|---|---|
| Subtract mean | ✅ | ❌ |
| Divide by std dev | ✅ | ❌ |
| Divide by RMS | ❌ | ✅ |
| Modern LLMs | early | **mainstream** (Llama/Qwen)|

**In net.h**: each layer has **two** RMSNorms — one before attention, one before MLP.

```c
// lines 48-49
float *rms_att_weight;   // normalization weight before attention (24, 896)
float *rms_ffn_weight;   // normalization weight before MLP      (24, 896)
// line 66
float *rms_final_weight; // final normalization after all layers (896,)
```

> Why two per layer? Because a layer has two sub-modules — Attention and MLP — each doing matrix multiplication, so each matrix multiplication needs a normalization beforehand. See Chapter 5, Section 5.2.

---

#### rms_eps (the small constant in normalization)

**What it is**: the ε in the denominator of RMSNorm, to prevent division by zero. `rms_eps = 1e-6`.

**Plain English**: in case the vector is all zeros, RMS = 0, and dividing by zero gives infinity. Add a tiny number as a safety net.

```c
float inv = 1.0f / sqrtf(mean(x²) + 1e-6);   // ← the 1e-6 here
```

---

### Group 5: Feed-Forward Network (MLP / FFN)

#### MLP / FFN (Multi-Layer Perceptron / Feed-Forward Network)

**What it is**: in each Transformer layer, a "two-layer fully connected network" that comes after attention.

**Plain English**: attention handles "exchanging information between tokens"; MLP handles "each token processing information on its own."
Attention is "a meeting to discuss"; MLP is "going back to your desk to think independently."

**Structure**: expand → activate → compress

```
896-dim ──→ 4864-dim ──→ activate (nonlinear) ──→ 896-dim
           expand        SwiGLU              compress
```

> MLP accounts for 88% of each layer's parameters. If you want to compress a model, MLP is the primary target (also why MoE reforms it).

---

#### SwiGLU (Swish-Gated Linear Unit)

**What it is**: the activation function Qwen uses. `output = silu(gate × x) * (up × x)`.

**Plain English**: a traditional activation is "one input passes through one function." SwiGLU adds a "gating" branch:
the gate decides **which channels to activate**, the up-projection provides **candidate values**, and the two are multiplied.

```
         ┌── @ W_gate → gate ── silu() ──┐
x (896-dim) ┤                              × (element-wise multiply) → 4864-dim → @ W_down → 896-dim
         └── @ W_up   → up   ────────────┘
```

**Why it beats ReLU**:

- ReLU is a hard gate (negatives go straight to zero, positives pass through unchanged)
- SwiGLU is a soft gate (smoothly controlled by a sigmoid, and the gate value is dynamically learned)

**The three weight matrices**:

```c
// net.h lines 62-64
float *w_gate; // gate:      (24, 4864, 896) decides "how much to open"
float *w_up;   // up-proj:   (24, 4864, 896) provides "candidate values"
float *w_down; // down-proj: (24, 896, 4864) compresses back to 896-dim
```

---

#### silu (SiLU / Swish activation function)

**What it is**: `silu(x) = x × sigmoid(x) = x / (1 + e^(-x))`.

**Plain English**: smoother than ReLU. Negatives don't snap to zero but slowly approach zero; positives are approximately the identity.

```
ReLU(x):  negatives → 0, positives → original   (edgy, gradients can break)
silu(x):  negatives → smoothly approach 0        (smooth, differentiable everywhere)
          positives → approximately x
```

---

### Group 6: Positional Encoding (RoPE)

#### RoPE (Rotary Position Embedding)

**What it is**: encodes positional information by "rotating" certain dimensions of the vector. `rope_theta = 1,000,000`.

**Plain English**: attention itself doesn't know the order of tokens ("cat bites person" and "person bites cat" look the same to it).
RoPE rotates each position by a different angle so that attention can sense "how far apart two tokens are."

```
position 0's vector: no rotation
position 1's vector: rotate a bit
position 2's vector: rotate twice as much
...

the attention score of two tokens = the difference of their rotation angles = relative position
```

**What rope_theta does**: the base frequency. The larger it is, the farther apart positions it can encode (supports longer context).

- GPT-2: theta=10000 (context 1024)
- **Qwen2.5: theta=1000000** (context 32768)

**In net.h**:

```c
// line 32
float rope_theta; // 1,000,000
```

**It only acts on Q and K**, not on V — positional information is only needed when computing relevance (Q·K).

---

### Group 7: KV Cache (Generation Acceleration)

#### KV Cache (Key-Value Cache)

**What it is**: stores the already-computed Key/Value of every token so it can be reused directly next time.

**Plain English**: when generating the N-th word, the K/V of the first N-1 words have already been computed — no need to recompute them.
Like cooking — prepped ingredients go in the fridge, ready to use next time, no need to re-prep.

```
Without KV Cache: generating the 5th word, must recompute K/V of the first 4 words → O(N²)
With KV Cache:    generating the 5th word, only compute the 5th word's K/V         → O(N)
                  the first 4's K/V are read from the cache
```

**In net.h**:

```c
// lines 85-86
float *key_cache;   // (24 layers, 2048 positions, 128-dim) stores all historical K
float *value_cache; // (24 layers, 2048 positions, 128-dim) stores all historical V
```

**Memory**: 2 × 24 × 2048 × 128 × 4 bytes = 50 MB.
Without GQA (using 14 heads), it would be 353 MB — that's the value of GQA.

---

#### seq_len (sequence length)

**What it is**: the maximum number of tokens we support. `seq_len = 2048` (set in code; the model itself supports 32768).

**Plain English**: how many tokens a conversation can hold. The KV Cache size is directly determined by this.
Set it smaller to save memory, larger to handle longer texts.

---

### Group 8: Other Concepts

#### Residual Connection

**What it is**: each layer's output = input + that layer's computed result. `x = x + layer(x)`.

**Plain English**: each layer doesn't directly replace the input; it "overlays" a small correction on top of the input.
Like grading homework — not rewriting it, but adding annotations onto the original.

```c
// in net.c there are two residuals per layer
s->x[i] += attention_out;   // attention result added back to the residual
s->x[i] += mlp_out;         // MLP result added back to the residual
```

**Why it's needed**: without residuals, deep networks (24 layers) suffer vanishing gradients and can't train.
Residuals let information "skip over" certain layers, and gradients can flow back directly.

---

#### Forward Pass (Forward)

**What it is**: given an input, compute from the first layer to the last to get the output.

**Plain English**: the process of the model "thinking" once. Feed in a token, compute all the way to logits.

```c
// net.h line 103
void forward(Config *config, TransformerWeights *w, RunState *s,
             int token, int pos);
```

**Contrast**: during training there's also a "backward pass" (backward) that computes gradients and updates weights.
**An inference engine only needs the forward pass** — which is why it's much simpler than training. See Chapter 5.

---

#### bf16 → fp32 (data type conversion)

**What it is**: weights are stored in bfloat16 (saves space) and computed in float32 (higher precision).

**Plain English**: the model is saved to disk in a "compressed format" (bf16, 2 bytes per number),
and decompressed into the "original format" (fp32, 4 bytes per number) for computation.

```
bf16: 16 bits = 1 sign + 8 exponent + 7 mantissa    (saves space, enough precision)
fp32: 32 bits = 1 sign + 8 exponent + 23 mantissa   (high precision, twice the space)

conversion: bf16 left-shifted by 16 bits = the high 16 bits of fp32 (they share the same exponent format)
```

See Chapter 2 (safetensors loading), Section 2.3.

---

## A Diagram Tying Together Every Layer's Parameters

```
input x (896-dim)
  │
  ├─ normalize rms_att_weight (896) ──→ adjust the value scale (prevent explosion)
  │        ↓
  │  Q/K/V projection wq/wk/wv + bq/bk/bv ──→ compute three roles (bias helps translate)
  │        ↓
  │     RoPE rotation (only on Q/K) ──→ encode position
  │        ↓
  │     store K/V into the KV Cache
  │        ↓
  │     Attention computation (14 Q heads attend over 2 KV heads, GQA)
  │        ↓
  │  O projection wo (no bias) ──→ merge the 14 heads
  │        ↓
  ├─ residual connection (x += attention_out)
  │        ↓
  ├─ normalize rms_ffn_weight (896) ──→ adjust the values once more
  │        ↓
  │  MLP w_gate + w_up ──→ expand to 4864-dim
  │     silu(gate) × up  (SwiGLU gating)
  │     w_down ──→ compress back to 896-dim
  │        ↓
  └─ residual connection (x += mlp_out)
       ↓
     output to the next layer (repeat 24 times)
       ↓
     final normalization rms_final_weight + project back to the vocabulary → logits
```

Every arrow corresponds to a segment of computation in `net.c`. For exactly how each operator computes, see Chapter 5.

---

## Study Suggestions

1. **Understand three numbers first**: `dim=896` (how long a word vector is), `n_layers=24` (how many layers),
   `vocab_size=151936` (how many words the model knows). The other numbers are all derived from these three.

2. **Sort out two lines**:
   - **The data line** (RunState): how a single token flows through the network
   - **The weight line** (TransformerWeights): which learnable parameters the network has stored

3. **Verify by hand**: run `./run fwd model.safetensors 198 0 -v` once,
   and match the intermediate values in the log against which step each term corresponds to.

4. **Compare with config.json**: the Config struct in `net.h` is just the C-language mapping of config.json —
   they correspond one to one.

---

> In the next chapter we enter the engine's core — the forward pass in `net.c`. Every operator (RMSNorm, matmul,
> RoPE, Attention, SwiGLU) will be broken down line by line, with real numbers running the whole flow.
> [Continue to Chapter 5 →](./05-forward.md)
