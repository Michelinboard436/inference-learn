/* net.c — Transformer 前向传播
 *
 * 这是引擎的核心。所有算子都用最朴素的实现，目的是"读得懂"，不是"跑得快"。
 * 真实引擎(llama.cpp 等)的优化点是: 量化 + SIMD + 重排内存布局 + 并行。
 *
 * Qwen2.5 单层前向流:
 *   x_norm = rmsnorm(x, rms_att_w)
 *   q,k,v = x_norm @ Wq/Wk/Wv + bq/bk/bv        # ★ Qwen 的 QKV 有 bias
 *   q,k  = rope(q,k, pos)                        # 只对 q,k 加旋转
 *   把 k,v 存进 KV cache (第 pos 列)
 *   att[h] = softmax(q[h] · k_cache[t] / sqrt(d))   # GQA: 2 个 kv 头共享给 14 个 q 头
 *   out = sum_t att[t] * v_cache[t]
 *   x = x + out @ Wo
 *   x_norm = rmsnorm(x, rms_ffn_w)
 *   x = x + (silu(x_norm @ Wgate) * (x_norm @ Wup)) @ Wdown   # SwiGLU
 *
 * 最后一层后:
 *   x = rmsnorm(x, rms_final_w)
 *   logits = x @ token_embedding^T                  # tie_word_embeddings
 */

#include "net.h"
#include "trace.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- 调试: dump 中间张量 ---- */
/* 设为 1 时 forward 会在第 0 层关键位置打印前 8 个 float, 供和 PyTorch 对照 */
int net_debug_dump = 0;

static void dump_vec(const char *name, const float *v, int n) {
    int k = n < 8 ? n : 8;
    printf("%s[0:%d] = [", name, k);
    for (int i = 0; i < k; i++) {
        printf("%.6f%s", v[i], i+1<k ? ", " : "");
    }
    printf("]\n");
    fflush(stdout);
}

/* ---- 工作内存分配 / 释放 ---- */

void malloc_run_state(RunState *s, Config *p) {
    int dim   = p->dim;
    int hd    = p->hidden_dim;
    int kvd   = p->n_kv_heads * p->head_dim;   /* 128 */
    int voc   = p->vocab_size;
    int seq   = p->seq_len;
    int nhead = p->n_heads;
    int L     = p->n_layers;

    s->x      = calloc(dim, sizeof(float));
    s->xb     = calloc(dim, sizeof(float));
    s->xb2    = calloc(dim, sizeof(float));
    s->hb     = calloc(hd,  sizeof(float));
    s->hb2    = calloc(hd,  sizeof(float));
    s->q      = calloc(dim, sizeof(float));
    s->k      = calloc(kvd, sizeof(float));
    s->v      = calloc(kvd, sizeof(float));
    s->att    = calloc((size_t)nhead * seq, sizeof(float));
    s->logits = calloc(voc,  sizeof(float));
    /* KV cache: 每层每个位置一份 K/V，每个大小 kvd */
    s->key_cache   = calloc((size_t)L * seq * kvd, sizeof(float));
    s->value_cache = calloc((size_t)L * seq * kvd, sizeof(float));

    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 ||
        !s->q || !s->k || !s->v || !s->att || !s->logits ||
        !s->key_cache || !s->value_cache) {
        fprintf(stderr, "RunState 分配失败 (内存不足?)\n");
        exit(1);
    }
}

void free_run_state(RunState *s) {
    free(s->x); free(s->xb); free(s->xb2);
    free(s->hb); free(s->hb2);
    free(s->q); free(s->k); free(s->v);
    free(s->att); free(s->logits);
    free(s->key_cache); free(s->value_cache);
    /* 避免悬空指针 */
    s->x = s->xb = s->xb2 = s->hb = s->hb2 = NULL;
    s->q = s->k = s->v = s->att = s->logits = NULL;
    s->key_cache = s->value_cache = NULL;
}

/* ============ 算子 ============ */

/* RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
 *
 * 为什么用 RMSNorm 而不是 LayerNorm:
 *   LayerNorm = 减均值 + 除标准差 + 乘缩放
 *   RMSNorm   = 除 RMS(均方根) + 乘缩放
 *   省掉了"减均值"这一步，计算量更小，经验上效果几乎一样。
 *   现代 LLM (Llama / Qwen) 基本都用 RMSNorm。
 */
static void rmsnorm(float *out, const float *x, const float *weight,
                    int size, float eps) {
    double ss = 0.0;
    for (int i = 0; i < size; i++) ss += (double)x[i] * x[i];
    ss = ss / size;
    ss += eps;
    float inv = 1.0f / sqrtf((float)ss);
    for (int i = 0; i < size; i++) {
        out[i] = (x[i] * inv) * weight[i];
    }
}

/* 最朴素的矩阵向量乘: out[d] = W[d,n] @ x[n] (+ bias[d])
 * W 以 row-major 存储: W[i*n + j] 是第 i 行第 j 列
 * 这是整个引擎最耗时的部分(占 ~95% 计算)。真实引擎会用 BLAS/SIMD 加速。
 */
static void matmul(float *out, const float *W, const float *x,
                   const float *bias, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = (bias ? bias[i] : 0.0f);
        const float *wrow = W + (size_t)i * n;
        for (int j = 0; j < n; j++) {
            val += wrow[j] * x[j];
        }
        out[i] = val;
    }
}

/* RoPE: 旋转位置编码
 *
 * 核心思想: 把 q(或 k) 的相邻两维看作一个复数 (a+bi)，
 * 乘以 e^{i*m*theta_k} 做旋转。不同位置 m 旋转角度不同，
 * 于是内积 q·k 自然带上了"相对位置"信息。
 *
 * theta_k = base^(-2k/head_dim), base=rope_theta(=1e6 for Qwen2.5)
 * base 越大，相邻维度频率差异越小，模型越能"看远"(支持更长上下文)。
 *
 * head_dim 必须是偶数(我们的是 64)。我们把相邻两维配对 (2i, 2i+1)。
 * (HF 实现用的是 "half" 模式：前一半和后一半配对，不是相邻配对。
 *  我们按 HF 的方式实现。)
 */
static void rope(float *vec, int head_dim, int pos, float base) {
    /* vec 指向某一个 head 的 q 或 k，长度 head_dim */
    for (int i = 0; i < head_dim / 2; i++) {
        /* 第 i 对的频率 */
        float freq = powf(base, (float)(-2*i) / (float)head_dim);
        /* 旋转角度 = 位置 × 频率 */
        float angle = pos * freq;
        float c = cosf(angle);
        float s = sinf(angle);
        /* HF "half" 风格: (vec[i], vec[i + head_dim/2]) 配对 */
        float a = vec[i];
        float b = vec[i + head_dim / 2];
        vec[i]               = a * c - b * s;
        vec[i + head_dim / 2] = a * s + b * c;
    }
}

/* softmax: 把 vec[0..n] 归一化成概率分布(数值稳定版)
 * 先减最大值，再 exp 再除以和。
 */
static void softmax(float *vec, int n) {
    float maxv = vec[0];
    for (int i = 1; i < n; i++) if (vec[i] > maxv) maxv = vec[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        vec[i] = expf(vec[i] - maxv);
        sum += vec[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) vec[i] *= inv;
}

/* SiLU (Swish) 激活: silu(x) = x * sigmoid(x) = x / (1+e^-x)
 * 比 ReLU 平滑，导数不为零，训练更稳。
 */
static float silu(float x) {
    return x / (1.0f + expf(-x));
}

/* ============ 完整前向传播 ============
 *
 * 给定输入 token 和位置 pos，跑一遍网络，结果写到 s->logits。
 */
void forward(Config *config, TransformerWeights *w, RunState *s,
             int token, int pos) {

    int dim      = config->dim;
    int hd       = config->hidden_dim;
    int L        = config->n_layers;
    int nhead    = config->n_heads;
    int nkvhead  = config->n_kv_heads;
    int d        = config->head_dim;
    int kvD      = nkvhead * d;            /* 2*64 = 128 */
    float theta  = config->rope_theta;
    float eps    = config->rms_eps;
    /* GQA: 每个 kv 头对应多少个 q 头。Qwen2.5-0.5B: 14/2 = 7 */
    int group    = nhead / nkvhead;

    /* 1) 取 token 的 embedding 作为初始 x
     * [模型结构] 起点要查表, 是 embedding 机制规定的
     * [模型数据] token_embedding_table 来自 model.embed_tokens.weight (151936, 896)
     * [程序]     按 token id × dim 定位到那一行, memcpy 拷出 896 个数
     * [文档]     BOOK.md 2.5 节, 4.2 节
     * [注意]     这 896 个数是固化的起点, 之后每层都会改写它 */
    float *embed_row = w->token_embedding_table + (size_t)token * dim;
    memcpy(s->x, embed_row, dim * sizeof(float));
    if (net_debug_dump) dump_vec("embed", s->x, dim);
    LOGD_L(CAT_KEY, 0, "forward: token=%d pos=%d  |x|=%.4f", token, pos, vec_norm(s->x, dim));
    log_vec(LOG_TRACE, "  embed", s->x, dim, 8);

    /* 2) 逐层 Transformer */
    for (int l = 0; l < L; l++) {
        LOGD_L(CAT_LAYER, 0, "┌─ Layer %d/%d", l, L);
        float xnorm_pre = vec_norm(s->x, dim);

        /* ---- 2a) attention 前的 RMSNorm (归一化 ①) ----
         * [模型结构] 这一步要做、用 RMSNorm 公式, 是 Qwen 架构规定的
         * [模型数据] w->rms_att_weight + l*dim 是这层的 896 个缩放系数
         *            (训练学的, 来自 safetensors 的 input_layernorm.weight)
         * [程序]     rmsnorm() 函数照公式执行, 不决定"做不做"
         * [文档]     BOOK.md 5.2 节, 参数清单 ①
         * [为什么]   防止数值经过 24 层矩阵乘法后爆炸 */
        rmsnorm(s->xb, s->x, w->rms_att_weight + (size_t)l * dim, dim, eps);
        LOGD_L(CAT_OP, 1, "RMSNorm(att): |x|=%.4f → |x'|=%.4f",
             xnorm_pre, vec_norm(s->xb, dim));
        log_vec(LOG_TRACE, "  norm_att", s->xb, dim, 8);

        /* ---- 2b) 计算 q,k,v (QKV 投影) ----
         * [模型结构] 要投影出 Q/K/V 三个角色, 是 attention 机制的核心
         * [模型数据] 下面 6 个权重全部训练学, 来自 safetensors:
         *            wq/wk/wv = 投影矩阵, bq/bk/bv = 偏置(Qwen 独有)
         * [程序]     matmul() 照公式 W@x+b 执行
         * [文档]     BOOK.md 5.3 节 matmul, 参数清单 ②③
         *
         * 取这层的权重起点 (每层独立一组):
         *   wq + l*dim*dim = 跳过前 l 层, 定位到第 l 层的 Wq */
        float *wq = w->wq + (size_t)l * dim * dim;     /* Q 投影权重  (896,896) 来自 self_attn.q_proj.weight */
        float *wk = w->wk + (size_t)l * kvD * dim;     /* K 投影权重  (128,896) 来自 self_attn.k_proj.weight, GQA 让它小 */
        float *wv = w->wv + (size_t)l * kvD * dim;     /* V 投影权重  (128,896) 来自 self_attn.v_proj.weight, 同上 */
        float *bq = w->bq + (size_t)l * dim;           /* Q 偏置      (896)    来自 self_attn.q_proj.bias, Qwen 独有 */
        float *bk = w->bk + (size_t)l * kvD;           /* K 偏置      (128)    来自 self_attn.k_proj.bias, Qwen 独有 */
        float *bv = w->bv + (size_t)l * kvD;           /* V 偏置      (128)    来自 self_attn.v_proj.bias, Qwen 独有 */

        /* Q = xb @ Wq + bq : 算出"我想找什么" (896维=14头×64)
         * K = xb @ Wk + bk : 算出"我是什么标签" (128维=2头×64, GQA)
         * V = xb @ Wv + bv : 算出"我携带什么内容" (128维)
         * matmul 参数: (out, W, x, bias, n=输入维, d=输出维), W 形状 (d, n) */
        matmul(s->q, wq, s->xb, bq, dim, dim);
        matmul(s->k, wk, s->xb, bk, dim, kvD);
        matmul(s->v, wv, s->xb, bv, dim, kvD);

        LOGD_L(CAT_OP, 1, "QKV投影: |Q|=%.3f |K|=%.3f |V|=%.3f  (Qwen QKV 有bias)",
             vec_norm(s->q, dim), vec_norm(s->k, kvD), vec_norm(s->v, kvD));
        log_vec(LOG_TRACE, "  q_proj", s->q, dim, 8);
        log_vec(LOG_TRACE, "  k_proj", s->k, kvD, 8);

        if (net_debug_dump && l == 0) {
            dump_vec("L0_rmsnorm_in", s->xb, dim);   /* rmsnorm 后=attention输入 */
            dump_vec("L0_q_proj", s->q, dim);        /* rope 前 */
            dump_vec("L0_k_proj", s->k, kvD);
        }

        /* ---- 2c) 对 q 和 k 的每个头应用 RoPE ---- */
        /* q 有 nhead 个头，每个 head_dim；k 有 nkvhead 个头 */
        for (int h = 0; h < nhead; h++) {
            rope(s->q + h * d, d, pos, theta);
        }
        for (int h = 0; h < nkvhead; h++) {
            rope(s->k + h * d, d, pos, theta);
        }
        /* RoPE 旋转角度: 第一个 head 的前 4 对的角度 */
        if (pos == 0) {
            LOGT_L(CAT_OP, 1, "RoPE: pos=0 时所有旋转角度为0 (cos=1,sin=0), q 不变");
        }
        log_vec(LOG_TRACE, "  q_rope", s->q, dim, 8);

        if (net_debug_dump && l == 0) {
            dump_vec("L0_q_after_rope", s->q, dim);
        }

        /* ---- 2d) 把当前 k,v 写进 KV cache 的第 pos 列 ---- */
        /* key_cache: 形状 (L, seq_len, kvD)。当前层 l 的第 pos 个位置 */
        float *krow = s->key_cache   + ((size_t)l * config->seq_len + pos) * kvD;
        float *vrow = s->value_cache + ((size_t)l * config->seq_len + pos) * kvD;
        memcpy(krow, s->k, kvD * sizeof(float));
        memcpy(vrow, s->v, kvD * sizeof(float));
        LOGD_L(CAT_OP, 1, "KV Cache: 写入第 %d 个位置 (当前缓存长度=%d)", pos, pos + 1);

        /* ---- 2e) 多头注意力 (含 GQA) ---- */
        /* 输出先存到 xb (后面加回残差) */
        /* 对每个 q 头 h:
         *   它对应的 kv 头是 h / group  (GQA 共享)
         *   att[h] = q[h] · k_cache[t] / sqrt(d),  t = 0..pos
         *   out = sum_t softmax(att)[t] * v_cache[t]
         */
        LOGD_L(CAT_OP, 1, "Attention: %d个Q头 × %d个KV头 (GQA: 每%d个Q共享1个KV, √d=√%d=%.2f)",
             nhead, nkvhead, group, d, sqrtf((float)d));
        for (int h = 0; h < nhead; h++) {
            int kvh = h / group;   /* GQA: 7 个 q 头共享 1 个 kv 头 */
            float *qh = s->q + h * d;

            /* 计算这个头对所有历史位置的注意力分数 */
            float *att = s->att + (size_t)h * config->seq_len;
            for (int t = 0; t <= pos; t++) {
                float *kt = s->key_cache + ((size_t)l * config->seq_len + t) * kvD
                            + kvh * d;   /* 取对应 kv 头的那段 d=64 维 */
                float score = 0.0f;
                for (int i = 0; i < d; i++) score += qh[i] * kt[i];
                score /= sqrtf((float)d);
                att[t] = score;
            }

            /* softmax over [0..pos] */
            softmax(att, pos + 1);

            /* 找出这个 head 最关注的位置 (top-3), 供 debug 级展示 */
            if (l < 3 || l == L - 1) {   /* 只对前几层和最后一层详细打, 避免太多 */
                int top_t[3] = {-1,-1,-1};
                float top_v[3] = {-1e30f,-1e30f,-1e30f};
                int ntop = pos + 1 < 3 ? pos + 1 : 3;   /* 实际能取几个 */
                for (int t = 0; t <= pos; t++) {
                    for (int r = 0; r < ntop; r++) {
                        if (att[t] > top_v[r]) {
                            for (int m = ntop - 1; m > r; m--) {
                                top_v[m] = top_v[m-1]; top_t[m] = top_t[m-1];
                            }
                            top_v[r] = att[t]; top_t[r] = t; break;
                        }
                    }
                }
                if (ntop == 1) {
                    LOGD_L(CAT_ATTENTION, 2, "head %2d (→kv%d): pos=0(%.2f) [仅1个位置]",
                           h, kvh, top_v[0]);
                } else if (ntop == 2) {
                    LOGD_L(CAT_ATTENTION, 2, "head %2d (→kv%d): pos=%d(%.2f) pos=%d(%.2f)",
                           h, kvh, top_t[0], top_v[0], top_t[1], top_v[1]);
                } else {
                    LOGD_L(CAT_ATTENTION, 2, "head %2d (→kv%d): pos=%d(%.2f) pos=%d(%.2f) pos=%d(%.2f)",
                         h, kvh, top_t[0], top_v[0], top_t[1], top_v[1], top_t[2], top_v[2]);
                }
            }

            /* 加权求和 v_cache[t] -> xb[h*d .. (h+1)*d] */
            float *out_h = s->xb + h * d;
            for (int i = 0; i < d; i++) out_h[i] = 0.0f;
            for (int t = 0; t <= pos; t++) {
                float *vt = s->value_cache + ((size_t)l * config->seq_len + t) * kvD
                            + kvh * d;
                float a = att[t];
                for (int i = 0; i < d; i++) out_h[i] += a * vt[i];
            }
        }
        LOGD_L(CAT_OP, 1, "Attention输出 |out|=%.4f", vec_norm(s->xb, dim));

        /* ---- 2f) 输出投影 Wo + 残差 ----
         * [模型结构] 把 14 个 attention 头的结果整合回一个向量, Qwen 规定 O 投影无 bias
         * [模型数据] w->wo 是这层的 O 投影权重 (896,896), 来自 self_attn.o_proj.weight
         * [程序]     matmul(..., NULL, ...) 第 4 个参数 NULL 表示无 bias
         * [文档]     BOOK.md 5.5 节, 参数清单 ④
         * [为什么]   14 头各算各的, 拼起来是杂乱的, Wo 重新整合 */
        float *wo = w->wo + (size_t)l * dim * dim;
        if (net_debug_dump && l == 0) {
            dump_vec("L0_o_proj_in", s->xb, dim);   /* attention 输出 = Wo 的输入 */
        }
        matmul(s->xb2, wo, s->xb, NULL, dim, dim);  /* Wo @ xb, 无 bias */
        for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];  /* 残差: x = x + Wo(attn_out) */
        LOGD_L(CAT_OP, 1, "Wo投影+残差1: |x|=%.4f → |x|=%.4f", xnorm_pre, vec_norm(s->x, dim));

        /* ---- 2g) MLP 前的 RMSNorm (归一化 ②) ----
         * [模型结构] 同一层里的第二次归一化, 在 MLP 前做 (Qwen 架构规定)
         * [模型数据] w->rms_ffn_weight 是另外 896 个缩放系数
         *            (和上面 rms_att_weight 不同, 来自 post_attention_layernorm.weight)
         * [为什么]   attention 算完数值又变了, 进 MLP 前必须再归一化一次
         * [文档]     BOOK.md 5.2 节 '每层有两个 RMSNorm' */
        float xn2 = vec_norm(s->x, dim);
        rmsnorm(s->xb, s->x, w->rms_ffn_weight + (size_t)l * dim, dim, eps);
        LOGD_L(CAT_OP, 1, "RMSNorm(ffn): |x|=%.4f → |x'|=%.4f", xn2, vec_norm(s->xb, dim));

        /* ---- 2h) SwiGLU MLP (占每层 88% 参数的大头) ----
         * [模型结构] 扩张→门控→压缩, 用 silu 做软门, 是 Qwen 架构规定的
         * [模型数据] 三个权重矩阵, 全部训练学, 来自 safetensors:
         *            w_gate/w_up/w_down 分别来自 mlp.gate_proj/up_proj/down_proj.weight
         * [程序]     三次 matmul + 一次逐元素 silu×up
         * [文档]     BOOK.md 5.7 节, 参数清单 ⑤
         * [为什么]   attention 管"词间交换信息", MLP 管"每个词独立加工",
         *            加工需要更大的空间, 所以扩张到 4864 维 */
        float *wg = w->w_gate + (size_t)l * hd * dim;   /* 门控权重 (4864,896) 算"开多少门" */
        float *wu = w->w_up   + (size_t)l * hd * dim;   /* 上投权重 (4864,896) 算"候选值" */
        float *wd = w->w_down + (size_t)l * dim * hd;   /* 下投权重 (896,4864) 压回 896 维 */

        matmul(s->hb,  wg, s->xb, NULL, dim, hd);   /* gate = xb @ W_gate  → 4864维 */
        matmul(s->hb2, wu, s->xb, NULL, dim, hd);   /* up   = xb @ W_up    → 4864维 */
        /* silu 前记录统计 (silu 会原地覆盖 s->hb) */
        float gate_act  = vec_active_ratio(s->hb, hd, 0.0f);
        float gate_norm = vec_norm(s->hb, hd);
        float up_norm   = vec_norm(s->hb2, hd);
        /* SwiGLU: 逐元素 silu(gate) × up, 门控决定放行多少候选值 */
        for (int i = 0; i < hd; i++) s->hb[i] = silu(s->hb[i]) * s->hb2[i];
        LOGD_L(CAT_OP, 1, "SwiGLU: gate激活率=%.0f%% |gate|=%.2f |up|=%.2f → |silu(g)*u|=%.2f",
             gate_act * 100, gate_norm, up_norm, vec_norm(s->hb, hd));

        /* 下投: 把 4864 维的思考结果压回主干道宽度 896 */
        matmul(s->xb2, wd, s->hb, NULL, hd, dim);
        /* 残差: x = x + MLP(x), 把加工结果叠加回去 */
        for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];
        LOGD_L(CAT_OP, 1, "Wdown+残差2: |x|=%.4f → |x|=%.4f", xn2, vec_norm(s->x, dim));

        if (net_debug_dump && l == 0) {
            dump_vec("L0_out", s->x, dim);   /* 第 0 层完整输出 */
        }
    }

    /* 3) 最终 RMSNorm (24 层全跑完后的最后一次归一化)
     * [模型数据] w->rms_final_weight 来自 model.norm.weight (896) */
    rmsnorm(s->x, s->x, w->rms_final_weight, dim, eps);
    if (net_debug_dump) dump_vec("final_norm", s->x, dim);
    LOGD_L(CAT_LAYER, 0, "└─ 最终RMSNorm: |x|=%.4f", vec_norm(s->x, dim));

    /* 4) 分类头: 算 logits (151936 个分数)
     * [模型结构] 用最终向量和每个词的 embedding 算点积(方向相似度)
     * [模型数据] 复用 token_embedding_table (tie_word_embeddings=true, 没有 lm_head)
     * [程序]     双重循环: 对每个词 v, 算 x·embed[v] 的点积
     * [文档]     BOOK.md 5.9 节
     * [本质]     在 896 维空间里找方向最接近的词 = 预测下一个词
     * logits[v] = sum_i x[i] * token_embedding[v][i] = x 和第 v 个词的点积 */
    for (int v = 0; v < config->vocab_size; v++) {
        float *embrow = w->token_embedding_table + (size_t)v * dim;
        float val = 0.0f;
        for (int i = 0; i < dim; i++) val += s->x[i] * embrow[i];
        s->logits[v] = val;
    }
    LOGD_L(CAT_LAYER, 0, "   logits: |.|=%.2f max=%.4f min=%.4f (tie_word_embeddings 复用 embed)",
         vec_norm(s->logits, config->vocab_size),
         vec_max(s->logits, config->vocab_size),
         vec_min(s->logits, config->vocab_size));
}
