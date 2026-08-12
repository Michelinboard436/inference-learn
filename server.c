/* server.c — 纯 C HTTP 服务器 (OpenAI 兼容 API)
 *
 * 实现一个最小的 HTTP 服务器, 支持 OpenAI /v1/chat/completions 端点。
 * 模型加载一次后常驻内存, 每个 HTTP 请求复用已加载的模型。
 *
 * 架构:
 *   1. 启动: 加载模型权重 + 分词器 → 监听端口
 *   2. 循环: accept 连接 → 读 HTTP → 解析 JSON → 跑生成 → 返回 JSON
 *   3. 单线程串行 (一个请求处理完才接下一个)
 *
 * 每个请求前清空 KV Cache, 保证请求之间不串数据。
 */

#include "server.h"
#include "net.h"
#include "safetensors.h"
#include "tokenizer.h"
#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

/* qwen2.5-0.5B 配置 (和 run.c 里的一致) */
static void server_set_config(Config *c, int seq_len) {
    c->dim        = 896;
    c->hidden_dim = 4864;
    c->n_layers   = 24;
    c->n_heads    = 14;
    c->n_kv_heads = 2;
    c->head_dim   = c->dim / c->n_heads;
    c->vocab_size = 151936;
    c->seq_len    = seq_len;
    c->rope_theta = 1000000.0f;
    c->rms_eps    = 1e-6f;
}

/* 模型全局状态 (常驻内存) */
typedef struct {
    Config              cfg;
    TransformerWeights  w;
    RunState            s;
    Tokenizer           tk;
    SafetensorsFile     st;
    int                 loaded;
} ServerState;

static ServerState g_state;

/* ============================================================
 * HTTP 底层: socket + 请求读取
 * ============================================================ */

/* 创建监听 socket, 返回 fd。失败返回 -1。 */
static int listen_on_port(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); return -1; }

    /* 允许端口复用 (重启时不用等 TIME_WAIT) */
    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(fd);
        return -1;
    }
    if (listen(fd, 8) < 0) {
        perror("listen");
        close(fd);
        return -1;
    }
    return fd;
}

/* 读取 HTTP 请求, 把 body 提取到 out_body。
 * 同时把请求路径提取到 out_path。
 * 返回 body 长度, 失败返回 -1。 */
static int read_http_request(int fd, char *out_path, int path_max,
                             char *out_body, int body_max) {
    char buf[65536];
    int total = 0;

    /* 读数据直到有完整的 HTTP 头 (\r\n\r\n) */
    while (total < (int)sizeof(buf) - 1) {
        int n = read(fd, buf + total, sizeof(buf) - 1 - total);
        if (n <= 0) break;
        total += n;
        buf[total] = 0;
        if (strstr(buf, "\r\n\r\n")) break;  /* 头读完了 */
    }
    buf[total] = 0;

    /* 解析请求行: GET /path HTTP/1.1 */
    if (sscanf(buf, "%*s %s", out_path) != 1) return -1;

    /* 找 Content-Length */
    int content_length = 0;
    char *cl = strstr(buf, "Content-Length:");
    if (!cl) cl = strstr(buf, "content-length:");
    if (cl) sscanf(cl + 15, "%d", &content_length);

    /* 提取 body (\r\n\r\n 之后的部分) */
    char *body_start = strstr(buf, "\r\n\r\n");
    int body_len = 0;
    if (body_start) {
        body_start += 4;
        body_len = total - (body_start - buf);

        /* 如果 body 没读完, 继续读 */
        while (body_len < content_length && body_len < body_max - 1) {
            int n = read(fd, out_body + body_len, content_length - body_len);
            if (n <= 0) break;
            body_len += n;
        }
        /* 先拷贝已读的部分 */
        int copy = body_len < body_max - 1 ? body_len : body_max - 1;
        memcpy(out_body, body_start, copy);
        out_body[copy] = 0;
        body_len = copy;
    } else {
        out_body[0] = 0;
    }

    return body_len;
}

/* 发送 HTTP 响应 */
static void send_response(int fd, int code, const char *status,
                          const char *content_type, const char *body) {
    char header[512];
    int body_len = strlen(body);
    int h = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %d\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "\r\n",
        code, status, content_type, body_len);
    write(fd, header, h);
    write(fd, body, body_len);
}

/* ============================================================
 * JSON 响应构造
 * ============================================================ */

/* 转义 JSON 字符串里的特殊字符 */
static void json_escape(const char *src, char *dst, int dst_max) {
    int j = 0;
    for (int i = 0; src[i] && j < dst_max - 8; i++) {
        char c = src[i];
        if (c == '"')       { dst[j++] = '\\'; dst[j++] = '"'; }
        else if (c == '\\') { dst[j++] = '\\'; dst[j++] = '\\'; }
        else if (c == '\n') { dst[j++] = '\\'; dst[j++] = 'n'; }
        else if (c == '\r') { dst[j++] = '\\'; dst[j++] = 'r'; }
        else if (c == '\t') { dst[j++] = '\\'; dst[j++] = 't'; }
        else if ((unsigned char)c < 0x20) { j += snprintf(dst+j, dst_max-j, "\\u%04x", c); }
        else { dst[j++] = c; }
    }
    dst[j] = 0;
}

/* 构造 OpenAI 格式的成功响应 */
static void build_chat_response(char *dst, int max,
                                const char *generated, int n_prompt, int n_generated) {
    char escaped[32768];
    json_escape(generated, escaped, sizeof(escaped));

    snprintf(dst, max,
        "{"
          "\"id\":\"chatcmpl-%d\","
          "\"object\":\"chat.completion\","
          "\"model\":\"qwen2.5-0.5b\","
          "\"choices\":[{"
            "\"index\":0,"
            "\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},"
            "\"finish_reason\":\"stop\""
          "}],"
          "\"usage\":{"
            "\"prompt_tokens\":%d,"
            "\"completion_tokens\":%d,"
            "\"total_tokens\":%d"
          "}"
        "}",
        (int)(time(NULL) % 100000), escaped,
        n_prompt, n_generated, n_prompt + n_generated);
}

/* 构造 /v1/models 响应 */
static void build_models_response(char *dst, int max) {
    snprintf(dst, max,
        "{\"object\":\"list\",\"data\":[{\"id\":\"qwen2.5-0.5b\",\"object\":\"model\",\"owned_by\":\"inference-learn\"}]}");
}

/* 构造错误响应 */
static void build_error_response(char *dst, int max, const char *message) {
    snprintf(dst, max,
        "{\"error\":{\"message\":\"%s\",\"type\":\"invalid_request_error\"}}", message);
}

/* ============================================================
 * 请求处理: 解析请求 → 生成 → 返回
 * ============================================================ */

/* 从 JSON body 提取最后一条 user message 的 content。
 * 返回提取的字符串 (指向 body 里的地址, 不需要 free), 或 NULL。 */
static const char *extract_prompt(const char *body) {
    JsonValue *root = json_parse(body);
    if (!root) return NULL;

    JsonValue *messages = json_object_get(root, "messages");
    if (!messages || messages->type != JSON_ARRAY) { json_free(root); return NULL; }

    /* 找最后一条 message, 提取 content */
    const char *result = NULL;
    int n = json_array_len(messages);
    for (int i = 0; i < n; i++) {
        JsonValue *msg = json_array_get(messages, i);
        JsonValue *content = json_object_get(msg, "content");
        if (content && content->type == JSON_STRING) {
            result = content->string;  /* 指向 JSON 树里的字符串 */
        }
    }

    /* 注意: root 不能 free, 因为 result 指向它内部的字符串。
       调用者用完后需要 free root。但这里简化处理: 不 free, 请求处理完会泄漏少量内存。
       (学习项目可接受; 生产环境需要更好的生命周期管理) */
    return result;
}

/* 从 JSON body 提取 temperature (找不到返回默认值) */
static float extract_temperature(const char *body, float default_val) {
    JsonValue *root = json_parse(body);
    if (!root) return default_val;
    JsonValue *t = json_object_get(root, "temperature");
    float result = (t && t->type == JSON_NUMBER) ? (float)t->number : default_val;
    json_free(root);
    return result;
}

/* 从 JSON body 提取 max_tokens (找不到返回默认值) */
static int extract_max_tokens(const char *body, int default_val) {
    JsonValue *root = json_parse(body);
    if (!root) return default_val;
    JsonValue *t = json_object_get(root, "max_tokens");
    int result = (t && t->type == JSON_NUMBER) ? (int)t->number : default_val;
    json_free(root);
    return result;
}

/* 处理一个聊天补全请求。
 * prompt: 用户输入文本
 * temperature / max_tokens / top_k: 采样参数
 * out: 输出缓冲区 (存放生成的文本)
 * 返回生成的 token 数 */
static int handle_completion(const char *prompt, float temperature,
                             int max_tokens, int top_k,
                             char *out, int out_max) {
    Config *cfg = &g_state.cfg;
    RunState *s = &g_state.s;
    Tokenizer *tk = &g_state.tk;

    /* 清空 KV Cache (每个请求从干净状态开始) */
    int kvd = cfg->n_kv_heads * cfg->head_dim;
    memset(s->key_cache, 0,
           (size_t)cfg->n_layers * cfg->seq_len * kvd * sizeof(float));
    memset(s->value_cache, 0,
           (size_t)cfg->n_layers * cfg->seq_len * kvd * sizeof(float));

    /* 编码 prompt */
    int prompt_tokens[1024];
    int n_prompt = tokenizer_encode(tk, prompt, -1, prompt_tokens, 1024);
    if (n_prompt == 0) { out[0] = 0; return 0; }

    /* 生成循环 (和 cmd_generate 的逻辑一样) */
    int total_len = n_prompt + max_tokens;
    if (total_len > cfg->seq_len) total_len = cfg->seq_len;

    /* 简单的 RNG (固定 seed, 因为服务器模式追求确定性) */
    unsigned int rng_state = 12345;

    int out_pos = 0;
    int token = prompt_tokens[0];
    int n_generated = 0;

    for (int pos = 0; pos < total_len; pos++) {
        forward(cfg, &g_state.w, s, token, pos);

        int next;
        if (pos < n_prompt - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            /* 贪心或采样 */
            if (temperature == 0.0f) {
                next = 0;
                float bestv = s->logits[0];
                for (int i = 1; i < cfg->vocab_size; i++) {
                    if (s->logits[i] > bestv) { bestv = s->logits[i]; next = i; }
                }
            } else {
                /* 简化采样: softmax + 轮盘赌 */
                float *probs = malloc(cfg->vocab_size * sizeof(float));
                if (!probs) break;
                float maxl = s->logits[0];
                for (int i = 1; i < cfg->vocab_size; i++)
                    if (s->logits[i] > maxl) maxl = s->logits[i];
                float inv_t = 1.0f / temperature;
                float sum = 0.0f;
                for (int i = 0; i < cfg->vocab_size; i++) {
                    probs[i] = expf((s->logits[i] - maxl) * inv_t);
                    sum += probs[i];
                }
                /* xorshift 简化版 */
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 17;
                rng_state ^= rng_state << 5;
                float r = (rng_state / 4294967296.0f) * sum;
                float cum = 0;
                next = cfg->vocab_size - 1;
                for (int i = 0; i < cfg->vocab_size; i++) {
                    cum += probs[i];
                    if (r < cum) { next = i; break; }
                }
                free(probs);
            }
        }

        if (next == tk->eos_id && pos >= n_prompt - 1) break;

        if (pos >= n_prompt - 1) {
            char buf[256];
            int blen = tokenizer_decode(tk, next, buf, sizeof(buf));
            if (out_pos + blen < out_max - 1) {
                memcpy(out + out_pos, buf, blen);
                out_pos += blen;
            }
            n_generated++;
        }
        token = next;
    }
    out[out_pos] = 0;
    return n_generated;
}

/* ============================================================
 * 主服务器循环
 * ============================================================ */

void run_server(const char *model_path, const char *tok_path,
                int port, float default_temp, int default_top_k) {
    /* 加载模型 (只加载一次) */
    fprintf(stderr, "[server] 加载模型: %s ...\n", model_path);
    server_set_config(&g_state.cfg, 2048);

    if (safetensors_open(model_path, &g_state.st) != 0) {
        fprintf(stderr, "[server] 无法打开 %s\n", model_path);
        return;
    }
    if (safetensors_load_weights(&g_state.st, &g_state.cfg, &g_state.w) != 0) {
        fprintf(stderr, "[server] 权重加载失败\n");
        return;
    }
    malloc_run_state(&g_state.s, &g_state.cfg);

    fprintf(stderr, "[server] 加载分词器: %s ...\n", tok_path);
    if (tokenizer_load(&g_state.tk, tok_path) != 0) {
        fprintf(stderr, "[server] 分词器加载失败\n");
        return;
    }

    g_state.loaded = 1;
    fprintf(stderr, "[server] 模型已就绪, 监听端口 %d ...\n", port);
    fprintf(stderr, "[server] 端点:\n");
    fprintf(stderr, "[server]   POST /v1/chat/completions  (OpenAI 兼容)\n");
    fprintf(stderr, "[server]   GET  /v1/models\n");
    fprintf(stderr, "[server] 按 Ctrl+C 退出\n\n");

    /* 忽略 SIGPIPE (客户端断开时 write 不会崩溃) */
    signal(SIGPIPE, SIG_IGN);

    /* 创建监听 socket */
    int listen_fd = listen_on_port(port);
    if (listen_fd < 0) return;

    /* accept 循环 */
    while (1) {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        int client_fd = accept(listen_fd, (struct sockaddr *)&client_addr, &addr_len);
        if (client_fd < 0) { perror("accept"); continue; }

        char client_ip[32];
        inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, sizeof(client_ip));

        /* 读 HTTP 请求 */
        char path[256];
        char body[65536];
        int body_len = read_http_request(client_fd, path, sizeof(path),
                                         body, sizeof(body));

        if (body_len < 0) {
            close(client_fd);
            continue;
        }

        /* 分发 */
        char response[65536];

        if (strcmp(path, "/v1/models") == 0) {
            /* GET /v1/models */
            build_models_response(response, sizeof(response));
            send_response(client_fd, 200, "OK", "application/json", response);
            fprintf(stderr, "[server] %s GET /v1/models → 200\n", client_ip);

        } else if (strcmp(path, "/v1/chat/completions") == 0 && body_len > 0) {
            /* POST /v1/chat/completions */
            const char *prompt = extract_prompt(body);
            if (!prompt) {
                build_error_response(response, sizeof(response),
                    "messages field is required");
                send_response(client_fd, 400, "Bad Request", "application/json", response);
                fprintf(stderr, "[server] %s POST /v1/chat/completions → 400 (no messages)\n", client_ip);
            } else {
                float temp = extract_temperature(body, default_temp);
                int max_tok = extract_max_tokens(body, 200);

                fprintf(stderr, "[server] %s POST /v1/chat/completions prompt=\"%s\" temp=%.1f ...\n",
                        client_ip, prompt, temp);

                char generated[32768];
                int n_generated = handle_completion(prompt, temp, max_tok,
                                                     default_top_k, generated,
                                                     sizeof(generated));

                /* 构造响应 */
                build_chat_response(response, sizeof(response),
                                    generated, 0, n_generated);
                send_response(client_fd, 200, "OK", "application/json", response);
                fprintf(stderr, "[server] %s → 200 (生成 %d token)\n",
                        client_ip, n_generated);
            }

        } else {
            /* 未知路径 */
            build_error_response(response, sizeof(response), "not found");
            send_response(client_fd, 404, "Not Found", "application/json", response);
            fprintf(stderr, "[server] %s %s → 404\n", client_ip, path);
        }

        close(client_fd);
    }

    /* 永远到不了这里 (Ctrl+C 退出) */
    close(listen_fd);
}
