/* run.c — 推理引擎入口
 *
 * 本文件结构 (从上到下):
 *   1. 辅助函数:   print_config / qwen25_0_5b_config
 *   2. 随机数:     rng_seed / rng_next / rng_uniform (xorshift128)
 *   3. 采样:       sample (greedy / temperature / top-k)
 *   4. 子命令:     cmd_fwd / cmd_info / cmd_encode / cmd_generate
 *   5. main:       参数解析 + 分发 (只有 ~30 行)
 *
 * 用法:
 *   ./run info <model.safetensors>              # 查看模型结构
 *   ./run encode <tokenizer.bin> "Hello world"   # 查看 BPE 分词
 *   ./run fwd <model.safetensors> <token_id>     # 单 token 前向 (验证)
 *   ./run <model.safetensors> <tokenizer.bin> "prompt"  # 生成文本
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "net.h"
#include "safetensors.h"
#include "json.h"
#include "tokenizer.h"
#include "trace.h"
#include "report.h"
#include "server.h"

/* ============================================================
 * 辅助函数
 * ============================================================ */

/* 打印 Qwen2.5-0.5B 的 config，方便对照 */
static void print_config(const Config *c) {
    printf("Config:\n");
    printf("  dim          = %d\n", c->dim);
    printf("  hidden_dim   = %d\n", c->hidden_dim);
    printf("  n_layers     = %d\n", c->n_layers);
    printf("  n_heads      = %d  (Q)\n", c->n_heads);
    printf("  n_kv_heads   = %d  (KV, GQA)\n", c->n_kv_heads);
    printf("  head_dim     = %d\n", c->head_dim);
    printf("  vocab_size   = %d\n", c->vocab_size);
    printf("  seq_len      = %d\n", c->seq_len);
    printf("  rope_theta   = %g\n", (double)c->rope_theta);
    printf("  rms_eps      = %g\n", (double)c->rms_eps);
}

/* 用 config.json 的硬编码值初始化 config (Qwen2.5-0.5B) */
static void qwen25_0_5b_config(Config *c, int seq_len) {
    c->dim        = 896;
    c->hidden_dim = 4864;
    c->n_layers   = 24;
    c->n_heads    = 14;
    c->n_kv_heads = 2;
    c->head_dim   = c->dim / c->n_heads;   /* 64 */
    c->vocab_size = 151936;
    c->seq_len    = seq_len;
    c->rope_theta = 1000000.0f;
    c->rms_eps    = 1e-6f;
}

/* ============================================================
 * 随机数生成器 (xorshift128)
 * ============================================================ */

typedef struct { unsigned int x,y,z,w; } RNG;

static void rng_seed(RNG *r, unsigned int s) {
    r->x = s ? s : 1;
    r->y = r->x * 1812433253u + 1;
    r->z = r->y * 1812433253u + 1;
    r->w = r->z * 1812433253u + 1;
}

static unsigned int rng_next(RNG *r) {
    unsigned int t = r->x ^ (r->x << 11);
    r->x = r->y; r->y = r->z; r->z = r->w;
    r->w = r->w ^ (r->w >> 19) ^ (t ^ (t >> 8));
    return r->w;
}

static float rng_uniform(RNG *r) {
    return (rng_next(r) >> 8) * (1.0f / 16777216.0f);
}

/* ============================================================
 * 采样: 从 logits 选出一个 token
 * ============================================================ */

static int sample(float *logits, int vocab_size,
                  float temperature, int top_k, RNG *rng) {
    /* 贪心: 直接 argmax */
    if (temperature == 0.0f) {
        int best = 0;
        float bestv = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > bestv) { bestv = logits[i]; best = i; }
        }
        return best;
    }

    LOGI_L(CAT_SAMPLE, 0, "采样: temperature=%.2f top_k=%d", temperature, top_k);

    /* 缩放 + softmax (在临时数组上操作, 不改原始 logits) */
    float *probs = malloc(vocab_size * sizeof(float));
    if (!probs) {
        fprintf(stderr, "sample: malloc 失败 (vocab_size=%d)\n", vocab_size);
        return 0;
    }
    float maxl = logits[0];
    for (int i = 1; i < vocab_size; i++) if (logits[i] > maxl) maxl = logits[i];
    float inv_t = 1.0f / temperature;
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = expf((logits[i] - maxl) * inv_t);
        sum += probs[i];
    }

    /* 打印 softmax 后的 top-5 概率 (debug 级) */
    {
        int top_id[5] = {-1,-1,-1,-1,-1};
        float top_p[5] = {-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
        for (int i = 0; i < vocab_size; i++) {
            for (int r = 0; r < 5; r++) {
                if (probs[i] > top_p[r]) {
                    for (int m = 4; m > r; m--) {
                        top_p[m] = top_p[m-1]; top_id[m] = top_id[m-1];
                    }
                    top_p[r] = probs[i]; top_id[r] = i; break;
                }
            }
        }
        LOGD_L(CAT_SAMPLE, 1, "softmax top-5: id=%d(%.1f%%) id=%d(%.1f%%) id=%d(%.1f%%) id=%d(%.1f%%) id=%d(%.1f%%)",
               top_id[0], top_p[0]/sum*100,
               top_id[1], top_p[1]/sum*100,
               top_id[2], top_p[2]/sum*100,
               top_id[3], top_p[3]/sum*100,
               top_id[4], top_p[4]/sum*100);
    }

    /* top_k 截断 */
    if (top_k > 0 && top_k < vocab_size) {
        char *keep = calloc(vocab_size, 1);
        if (!keep) { free(probs); return 0; }
        for (int s = 0; s < top_k; s++) {
            int best = -1; float bestv = -1e30f;
            for (int i = 0; i < vocab_size; i++) {
                if (!keep[i] && probs[i] > bestv) { bestv = probs[i]; best = i; }
            }
            if (best < 0) break;
            keep[best] = 1;
        }
        sum = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            if (!keep[i]) probs[i] = 0.0f;
            else sum += probs[i];
        }
        free(keep);
        LOGD_L(CAT_SAMPLE, 1, "top_k=%d 截断: 保留 %d 个候选, 其余清零, 重算总概率=%.4f",
               top_k, top_k, sum);
    }

    /* 轮盘赌采样 */
    float r = rng_uniform(rng) * sum;
    LOGD_L(CAT_SAMPLE, 1, "轮盘赌: 随机数×总概率 = %.4f × %.4f → 阈值=%.4f",
           r / sum, sum, r);
    float cum = 0.0f;
    int chosen = vocab_size - 1;
    for (int i = 0; i < vocab_size; i++) {
        cum += probs[i];
        if (r < cum) { chosen = i; break; }
    }
    LOGI_L(CAT_SAMPLE, 0, "  → 选中 id=%d (累积概率落在 %.4f 处)", chosen, r);
    free(probs);
    return chosen;
}

/* ============================================================
 * 子命令: fwd — 单 token 前向传播 (数值验证用)
 *
 *   ./run fwd model.safetensors <token_id> [pos] [dump]
 *   ./run fwd model.safetensors 198 0           # 198 = '\n'
 *
 * 加载权重, 输入一个 token, 跑一次 forward, 打印 top-5 logits。
 * 用来和 PyTorch 对比数值是否正确。
 * ============================================================ */
static int cmd_fwd(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "fwd 用法: %s fwd <model.safetensors> <token_id> [pos] [dump]\n", argv[0]);
        return 1;
    }
    Config cfg;
    qwen25_0_5b_config(&cfg, 2048);
    int token = atoi(argv[3]);
    int pos   = (argc >= 5) ? atoi(argv[4]) : 0;

    fprintf(stderr, "[加载] %s ...\n", argv[2]);
    SafetensorsFile st;
    if (safetensors_open(argv[2], &st) != 0) return 1;
    TransformerWeights w;
    if (safetensors_load_weights(&st, &cfg, &w) != 0) {
        fprintf(stderr, "权重加载失败\n"); return 1;
    }
    RunState s;
    malloc_run_state(&s, &cfg);

    /* 若额外参数是 "dump" 则打开逐层中间张量打印 */
    if (argc >= 6 && strcmp(argv[5], "dump") == 0) {
        net_debug_dump = 1;
    }

    fprintf(stderr, "[forward] token=%d pos=%d ...\n", token, pos);
    forward(&cfg, &w, &s, token, pos);

    /* 打印 logits 的 top-5 */
    int top[5]; float topv[5];
    for (int k = 0; k < 5; k++) { top[k] = -1; topv[k] = -1e30f; }
    for (int v = 0; v < cfg.vocab_size; v++) {
        float lv = s.logits[v];
        for (int k = 0; k < 5; k++) {
            if (lv > topv[k]) {
                for (int m = 4; m > k; m--) {
                    topv[m] = topv[m-1]; top[m] = top[m-1];
                }
                topv[k] = lv; top[k] = v;
                break;
            }
        }
    }
    printf("=== top-5 next-token (input token=%d, pos=%d) ===\n", token, pos);
    for (int k = 0; k < 5; k++) {
        printf("  id=%-8d logit=%.6f\n", top[k], topv[k]);
    }
    printf("  (参考) logits[151643(EOS)] = %.6f\n", s.logits[151643]);

    free_run_state(&s);
    safetensors_close(&st);
    return 0;
}

/* ============================================================
 * 子命令: info — 查看 safetensors 文件结构
 *
 *   ./run info model.safetensors
 *
 * 打印所有张量的名字、shape、dtype、字节偏移。
 * 并校验 Qwen 独有的 QKV bias 是否存在。
 * ============================================================ */
static int cmd_info(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "info 需要文件名: %s info <model.safetensors>\n", argv[0]);
        return 1;
    }
    Config cfg;
    qwen25_0_5b_config(&cfg, 2048);
    print_config(&cfg);

    SafetensorsFile st;
    if (safetensors_open(argv[2], &st) != 0) {
        fprintf(stderr, "无法打开 %s\n", argv[2]);
        return 1;
    }
    printf("\n=== safetensors 张量列表 ===\n");
    printf("header 长度: %lu 字节, raw buffer: %lu 字节\n",
           st.header_len, st.data_size);

    /* 遍历 header JSON, 打印每个张量 */
    JsonValue *root = json_parse(st.header_json);
    if (!root) {
        fprintf(stderr, "header JSON 解析失败\n");
        safetensors_close(&st);
        return 1;
    }
    int count = 0;
    for (JsonValue *c = root->first_child; c; c = c->next_sibling) {
        if (c->type != JSON_STRING) continue;
        const char *name = c->string;
        JsonValue *v = c->value_child;
        JsonValue *dtype = json_object_get(v, "dtype");
        JsonValue *shape = json_object_get(v, "shape");
        JsonValue *doff  = json_object_get(v, "data_offsets");

        printf("  %-55s dtype=%-5s shape=[", name,
               dtype && dtype->type == JSON_STRING ? dtype->string : "?");
        if (shape) {
            int n = json_array_len(shape);
            for (int i = 0; i < n; i++) {
                printf("%lld%s", (long long)json_array_get(shape, i)->number,
                       i+1 < n ? "," : "");
            }
        }
        printf("]");
        if (doff) {
            unsigned long o0 = (unsigned long)json_array_get(doff, 0)->number;
            unsigned long o1 = (unsigned long)json_array_get(doff, 1)->number;
            printf("  data[%lu, %lu] (%lu bytes)", o0, o1, o1 - o0);
        }
        printf("\n");
        count++;
    }
    printf("共 %d 个张量\n", count);

    /* 关键校验: Qwen 的 QKV bias 是否存在 */
    unsigned long off, nb; int shp[4], ndim; const char *dt;
    const char *checks[] = {
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.self_attn.k_proj.bias",
        "model.layers.0.self_attn.v_proj.bias",
        "model.layers.0.self_attn.o_proj.bias",
        "lm_head.weight",
    };
    printf("\n=== 关键校验 ===\n");
    for (size_t i = 0; i < sizeof(checks)/sizeof(checks[0]); i++) {
        int found = (safetensors_find(&st, checks[i], &off, &nb, shp, &ndim, &dt) == 0);
        printf("  %-55s %s\n", checks[i], found ? "存在" : "不存在 (预期)");
    }

    json_free(root);
    safetensors_close(&st);
    return 0;
}

/* ============================================================
 * 子命令: encode — 查看 BPE 分词过程
 *
 *   ./run encode tokenizer.bin "Hello world"
 *
 * 打印每个 token 的 id + 解码后的字符串。
 * ============================================================ */
static int cmd_encode(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "encode 用法: %s encode <tokenizer.bin> \"<text>\"\n", argv[0]);
        return 1;
    }
    Tokenizer tk;
    if (tokenizer_load(&tk, argv[2]) != 0) {
        fprintf(stderr, "分词器加载失败\n"); return 1;
    }
    const char *text = argv[3];
    int tokens[1024];
    int n = tokenizer_encode(&tk, text, -1, tokens, 1024);
    printf("输入: %s\n", text);
    printf("编码出 %d 个 token:\n", n);
    char buf[256];
    for (int i = 0; i < n; i++) {
        int blen = tokenizer_decode(&tk, tokens[i], buf, sizeof(buf));
        printf("  [%3d] id=%-8d ", i, tokens[i]);
        for (int b = 0; b < blen; b++) {
            unsigned char c = (unsigned char)buf[b];
            if (c >= 32 && c < 127) printf("%c", c);
            else printf("\\x%02x", c);
        }
        printf("\n");
    }
    printf("ids = [");
    for (int i = 0; i < n; i++) printf("%d%s", tokens[i], i+1<n?",":"");
    printf("]\n");
    tokenizer_free(&tk);
    return 0;
}

/* ============================================================
 * 子命令: generate — 文本生成 (默认)
 *
 *   ./run model.safetensors tokenizer.bin "prompt" [-t temp] [-k topk] [-s seed]
 *
 * 流程: prompt → BPE 编码 → prefill (填 KV Cache) → decode (采样生成) → 解码打印
 * ============================================================ */
static int cmd_generate(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "需要 model + tokenizer + prompt\n");
        return 1;
    }
    const char *model_path = argv[1];
    const char *tok_path   = argv[2];
    const char *prompt     = argv[3];

    float temperature = 0.0f;
    int   top_k       = 40;
    int   seed        = 12345;

    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i+1 < argc) {
            temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-k") == 0 && i+1 < argc) {
            top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i+1 < argc) {
            seed = atoi(argv[++i]);
        }
    }

    /* 加载分词器, 编码 prompt */
    Tokenizer tk;
    if (tokenizer_load(&tk, tok_path) != 0) {
        fprintf(stderr, "分词器加载失败\n"); return 1;
    }
    int prompt_tokens[1024];
    int n_prompt = tokenizer_encode(&tk, prompt, -1, prompt_tokens, 1024);
    if (n_prompt == 0) {
        fprintf(stderr, "prompt 编码为空\n"); return 1;
    }
    LOGI("prompt 编码成 %d 个 token", n_prompt);

    /* 加载模型权重 */
    LOGI("加载权重: %s ...", model_path);
    SafetensorsFile st;
    if (safetensors_open(model_path, &st) != 0) return 1;
    Config cfg;
    qwen25_0_5b_config(&cfg, 2048);
    TransformerWeights w;
    if (safetensors_load_weights(&st, &cfg, &w) != 0) {
        fprintf(stderr, "权重加载失败\n"); return 1;
    }
    RunState s;
    malloc_run_state(&s, &cfg);

    RNG rng;
    rng_seed(&rng, (unsigned int)seed);

    /* 生成循环: prefill + decode */
    int max_new = 200;
    int total_len = n_prompt + max_new;
    if (total_len > cfg.seq_len) total_len = cfg.seq_len;

    char buf[256];
    printf("%s", prompt);
    fflush(stdout);

    LOGI_L(CAT_KEY, 0, "阶段 1: Prefill (处理 %d 个 prompt token)", n_prompt);

    int token = prompt_tokens[0];
    int n_generated = 0;
    for (int pos = 0; pos < total_len; pos++) {
        if (pos == n_prompt) {
            LOGI_L(CAT_KEY, 0, "阶段 2: Decode (自回归生成)");
        }
        if (pos >= n_prompt) {
            char decbuf[128];
            int dlen = tokenizer_decode(&tk, token, decbuf, sizeof(decbuf));
            LOGI_L(CAT_KEY, 0, "生成第 %d 个 token: id=%d \"%.*s\"",
                   n_generated + 1, token, dlen, decbuf);
        }

        if (trace_get_report() && pos >= n_prompt - 1) {
            trace_clear();
        }

        forward(&cfg, &w, &s, token, pos);

        int next;
        if (pos < n_prompt - 1) {
            next = prompt_tokens[pos + 1];
            LOGD_L(CAT_TEXT, 0, "prefill pos=%d: 用 prompt 的下一个 token id=%d", pos, next);
        } else {
            next = sample(s.logits, cfg.vocab_size, temperature, top_k, &rng);
        }

        if (next == tk.eos_id && pos >= n_prompt - 1) {
            LOGI("遇到 EOS, 停止生成");
            break;
        }

        if (pos >= n_prompt - 1) {
            int blen = tokenizer_decode(&tk, next, buf, sizeof(buf));
            fwrite(buf, 1, blen, stdout);
            fflush(stdout);
            n_generated++;
        }

        token = next;
    }
    printf("\n");

    LOGI("生成完成: 共 %d 个新 token", n_generated);

    /* 生成 HTML 报告 */
    if (trace_get_report()) {
        if (report_generate("trace_report.html") == 0) {
            LOGI("HTML 报告已生成: trace_report.html");
        }
    }

    /* 清理 */
    trace_free();
    free_run_state(&s);
    safetensors_close(&st);
    tokenizer_free(&tk);
    return 0;
}

/* ============================================================
 * 子命令: serve — 启动 HTTP 服务器 (OpenAI 兼容 API)
 *
 *   ./run serve model.safetensors tokenizer.bin [--port 8000]
 *
 * 模型加载一次后常驻内存, 通过 HTTP 接收请求。
 * 端点: POST /v1/chat/completions, GET /v1/models
 * ============================================================ */
static int cmd_serve(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "serve 用法: %s serve <model.safetensors> <tokenizer.bin> [--port 8000]\n", argv[0]);
        return 1;
    }
    int port = 8000;
    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--port") == 0 && i+1 < argc) {
            port = atoi(argv[++i]);
        }
    }
    run_server(argv[2], argv[3], port, 0.0f, 40);
    return 0;
}

/* ============================================================
 * main — 参数解析 + 子命令分发
 *
 * 只有这一个入口, 根据 argv[1] 分发到对应子命令函数。
 * 全局参数 (-v / -vv / --report) 在分发前预扫描。
 * ============================================================ */
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "用法:\n"
            "  %s info <model.safetensors>\n"
            "  %s encode <tokenizer.bin> \"<text>\"\n"
            "  %s fwd <model.safetensors> <token_id> [pos]\n"
            "  %s serve <model.safetensors> <tokenizer.bin> [--port 8000]\n"
            "  %s <model.safetensors> <tokenizer.bin> \"<prompt>\" "
            "[-v|-vv] [--report] [-t temp] [-k topk] [-s seed]\n",
            argv[0], argv[0], argv[0], argv[0], argv[0]);
        return 1;
    }

    /* 全局预扫描: -v / -vv / --report (所有子命令通用) */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0) {
            trace_set_level(LOG_DEBUG);
        } else if (strcmp(argv[i], "-vv") == 0) {
            trace_set_level(LOG_TRACE);
        } else if (strcmp(argv[i], "--report") == 0) {
            trace_set_report(1);
        }
    }
    if (trace_get_report() && trace_get_level() < LOG_TRACE) {
        trace_set_level(LOG_TRACE);
    }

    /* 分发到对应子命令 */
    if (strcmp(argv[1], "fwd") == 0)     return cmd_fwd(argc, argv);
    if (strcmp(argv[1], "info") == 0)    return cmd_info(argc, argv);
    if (strcmp(argv[1], "encode") == 0)  return cmd_encode(argc, argv);
    if (strcmp(argv[1], "serve") == 0)   return cmd_serve(argc, argv);
    return cmd_generate(argc, argv);  /* 默认: 生成模式 */
}
