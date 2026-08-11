#ifndef NET_H
#define NET_H

/*
 * net.h — 模型结构与前向传播
 *
 * 这里定义 Qwen2.5-0.5B 的三大结构体。所有维度的精确含义见 README。
 *
 * 对照 config.json:
 *   hidden_size        = dim        = 896
 *   num_hidden_layers  = n_layers   = 24
 *   num_attention_heads= n_heads    = 14
 *   num_key_value_heads= n_kv_heads = 2   (GQA: 每 7 个 Q 头共享 1 组 KV)
 *   head_dim           = dim/n_heads= 64  (RoPE 作用在它上面)
 *   intermediate_size  = hidden_dim = 4864
 *   vocab_size         = 151936
 *   rope_theta         = 1,000,000
 *   rms_norm_eps       = 1e-6
 *   tie_word_embeddings= true  → lm_head 复用 token_embedding，不单独存储
 */

/* 运行时配置：所有维度的"源头"。值由 config.json 填入。 */
typedef struct {
    int dim;          // 残差流宽度 896
    int hidden_dim;   // MLP 中间宽度 4864
    int n_layers;     // Transformer 层数 24
    int n_heads;      // Q 头数 14
    int n_kv_heads;   // KV 头数 2 (GQA)
    int head_dim;     // 每个头的维度 64 = dim / n_heads
    int vocab_size;   // 151936
    int seq_len;      // 我们支持的最大序列长度 (运行时决定，<=32768)
    float rope_theta; // 1,000,000
    float rms_eps;    // 1e-6
} Config;

/* 所有权重张量。连续存储，按层堆叠。
 * bf16 safetensors 加载时统一转成 fp32。
 *
 * ★ Qwen 特有点(和 Llama 的区别)：
 *   - Q/K/V 投影有 bias (bq/bk/bv)，O 投影没有 bias
 *   - 这是因为 Qwen 训练时在 QKV 上加了 bias，Llama 系列全部无 bias
 */
typedef struct {
    /* token embedding: 同时也是 lm_head (tie_word_embeddings=true) */
    float *token_embedding_table;  // (vocab_size, dim) = (151936, 896)

    /* 每层的权重，按层叠在一起方便索引 */
    float *rms_att_weight;   // (n_layers, dim)
    float *rms_ffn_weight;   // (n_layers, dim)

    float *wq;   // (n_layers, n_heads  * head_dim, dim) = (24, 896, 896)
    float *wk;   // (n_layers, n_kv_heads* head_dim, dim) = (24, 128, 896)
    float *wv;   // (n_layers, n_kv_heads* head_dim, dim) = (24, 128, 896)
    float *wo;   // (n_layers, n_heads  * head_dim, dim) = (24, 896, 896)

    float *bq;   // (n_layers, n_heads  * head_dim) = (24, 896)   ← Qwen 独有
    float *bk;   // (n_layers, n_kv_heads* head_dim) = (24, 128)  ← Qwen 独有
    float *bv;   // (n_layers, n_kv_heads* head_dim) = (24, 128)  ← Qwen 独有
    // 注意: wo 没有 bias (Qwen 设计如此)

    /* SwiGLU MLP: out = silu(gate @ x) * (up @ x), 然后 @ down */
    float *w_gate; // (n_layers, hidden_dim, dim) = (24, 4864, 896)
    float *w_up;   // (n_layers, hidden_dim, dim) = (24, 4864, 896)
    float *w_down; // (n_layers, dim, hidden_dim) = (24, 896, 4864)

    float *rms_final_weight; // (dim,) = (896,)
} TransformerWeights;

/* 前向传播用的工作内存。每次 forward 复用同一份。
 * KV cache 贯穿整个生成过程，存放历史 token 的 K/V。
 */
typedef struct {
    float *x;      // (dim,)          当前层输入
    float *xb;     // (dim,)          attention 内部的归一化后输入
    float *xb2;    // (dim,)          MLP 内部的归一化后输入(备份)
    float *hb;     // (hidden_dim,)   MLP gate/up 的输出
    float *hb2;    // (hidden_dim,)   MLP up 的输出
    float *q;      // (dim,)          当前 token 的 Q
    float *k;      // (n_kv_heads*head_dim,) 当前 token 的 K (= 128)
    float *v;      // (n_kv_heads*head_dim,) 当前 token 的 V (= 128)
    float *att;    // (n_heads, seq_len)  注意力分数
    float *logits; // (vocab_size,)   最终输出

    /* KV cache: 存所有历史位置的 K/V，attention 时复用 */
    float *key_cache;   // (n_layers, seq_len, n_kv_heads*head_dim)
    float *value_cache; // (n_layers, seq_len, n_kv_heads*head_dim)
} RunState;

/* ============ 公开 API ============ */

/* 调试开关: 设为 1 时 forward 会打印第 0 层的中间张量前 8 个元素 */
extern int net_debug_dump;

/* 从 config 初始化 RunState 的工作内存 */
void malloc_run_state(RunState *s, Config *p);

/* 释放 RunState */
void free_run_state(RunState *s);

/* 给定 token 和当前位置，跑一次前向，写 logits 到 s->logits。
 * pos: 这个 token 在序列中的位置 (0-based)，用来定位 KV cache 和 RoPE。
 */
void forward(Config *config, TransformerWeights *w, RunState *s,
             int token, int pos);

#endif // NET_H
