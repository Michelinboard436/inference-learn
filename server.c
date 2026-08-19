/* server.c — 纯 C HTTP 服务器 (OpenAI 兼容 API)
 *
 * 实现一个最小的 HTTP 服务器, 支持 OpenAI /v1/chat/completions 端点。
 * 模型加载一次后常驻内存, 每个 HTTP 请求复用已加载的模型。
 *
 * 功能:
 *   - OpenAI 兼容 API (/v1/chat/completions, /v1/models)
 *   - 多轮对话 + system prompt (Qwen chat 模板)
 *   - 流式响应 (SSE, stream: true)
 *   - 并发 (fork, 多请求并行处理)
 *
 * 架构:
 *   1. 启动: 加载模型权重 + 分词器 → 监听端口
 *   2. 循环: accept → fork → 子进程处理请求 → 父进程继续 accept
 *   3. 子进程: 读 HTTP → 解析 JSON → 跑生成 → 返回 JSON/SSE → exit
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
#include <sys/wait.h>
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

static int listen_on_port(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); return -1; }
    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind"); close(fd); return -1;
    }
    if (listen(fd, 8) < 0) {
        perror("listen"); close(fd); return -1;
    }
    return fd;
}

static int read_http_request(int fd, char *out_path, int path_max,
                             char *out_body, int body_max) {
    char buf[65536];
    int total = 0;
    while (total < (int)sizeof(buf) - 1) {
        int n = read(fd, buf + total, sizeof(buf) - 1 - total);
        if (n <= 0) break;
        total += n;
        buf[total] = 0;
        if (strstr(buf, "\r\n\r\n")) break;
    }
    buf[total] = 0;
    if (sscanf(buf, "%*s %s", out_path) != 1) return -1;

    int content_length = 0;
    char *cl = strstr(buf, "Content-Length:");
    if (!cl) cl = strstr(buf, "content-length:");
    if (cl) sscanf(cl + 15, "%d", &content_length);

    char *body_start = strstr(buf, "\r\n\r\n");
    int body_len = 0;
    if (body_start) {
        body_start += 4;
        body_len = total - (body_start - buf);
        while (body_len < content_length && body_len < body_max - 1) {
            int n = read(fd, out_body + body_len, content_length - body_len);
            if (n <= 0) break;
            body_len += n;
        }
        int copy = body_len < body_max - 1 ? body_len : body_max - 1;
        memcpy(out_body, body_start, copy);
        out_body[copy] = 0;
        body_len = copy;
    } else {
        out_body[0] = 0;
    }
    return body_len;
}

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

/* 发送 SSE 流式响应头 (不带 Content-Length, 连接保持打开) */
static void send_stream_headers(int fd) {
    const char *header =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: keep-alive\r\n"
        "\r\n";
    write(fd, header, strlen(header));
}

/* ============================================================
 * JSON 工具
 * ============================================================ */

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

static void build_chat_response(char *dst, int max,
                                const char *generated, int n_prompt, int n_generated) {
    char escaped[32768];
    json_escape(generated, escaped, sizeof(escaped));
    snprintf(dst, max,
        "{\"id\":\"chatcmpl-%d\",\"object\":\"chat.completion\",\"model\":\"qwen2.5-0.5b\","
        "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},\"finish_reason\":\"stop\"}],"
        "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
        (int)(time(NULL) % 100000), escaped, n_prompt, n_generated, n_prompt + n_generated);
}

static void build_models_response(char *dst, int max) {
    snprintf(dst, max,
        "{\"object\":\"list\",\"data\":[{\"id\":\"qwen2.5-0.5b\",\"object\":\"model\",\"owned_by\":\"inference-learn\"}]}");
}

static void build_error_response(char *dst, int max, const char *message) {
    snprintf(dst, max, "{\"error\":{\"message\":\"%s\",\"type\":\"invalid_request_error\"}}", message);
}

/* ============================================================
 * ★ 任务①: 多轮对话 + system prompt
 *
 * 从 messages 数组构建 Qwen chat 模板:
 *   <|im_start|>system\n{system}<|im_end|>\n
 *   <|im_start|>user\n{user}<|im_end|>\n
 *   <|im_start|>assistant\n{assistant}<|im_end|>\n
 *   <|im_start|>assistant\n   ← 模型从这里开始生成
 *
 * 返回 malloc 的字符串, 调用者负责 free。
 * ============================================================ */
static char *extract_chat_prompt(const char *body) {
    JsonValue *root = json_parse(body);
    if (!root) return NULL;

    JsonValue *messages = json_object_get(root, "messages");
    if (!messages || messages->type != JSON_ARRAY) { json_free(root); return NULL; }

    /* 构建完整的 chat 模板字符串 */
    char *prompt = malloc(32768);
    if (!prompt) { json_free(root); return NULL; }
    prompt[0] = 0;
    int pos = 0;

    int n = json_array_len(messages);
    for (int i = 0; i < n; i++) {
        JsonValue *msg = json_array_get(messages, i);
        JsonValue *role = json_object_get(msg, "role");
        JsonValue *content = json_object_get(msg, "content");
        if (!role || !content || role->type != JSON_STRING || content->type != JSON_STRING)
            continue;

        /* 拼接: <|im_start|>{role}\n{content}<|im_end|>\n */
        pos += snprintf(prompt + pos, 32768 - pos,
                        "<|im_start|>%s\n%s<|im_end|>\n",
                        role->string, content->string);
    }

    /* 最后追加 assistant 前缀, 模型从这里开始生成 */
    pos += snprintf(prompt + pos, 32768 - pos, "<|im_start|>assistant\n");

    json_free(root);
    return prompt;
}

static float extract_temperature(const char *body, float default_val) {
    JsonValue *root = json_parse(body);
    if (!root) return default_val;
    JsonValue *t = json_object_get(root, "temperature");
    float result = (t && t->type == JSON_NUMBER) ? (float)t->number : default_val;
    json_free(root);
    return result;
}

static int extract_max_tokens(const char *body, int default_val) {
    JsonValue *root = json_parse(body);
    if (!root) return default_val;
    JsonValue *t = json_object_get(root, "max_tokens");
    int result = (t && t->type == JSON_NUMBER) ? (int)t->number : default_val;
    json_free(root);
    return result;
}

/* ★ 任务②: 检查是否请求流式响应 */
static int extract_stream(const char *body) {
    JsonValue *root = json_parse(body);
    if (!root) return 0;
    JsonValue *s = json_object_get(root, "stream");
    int result = (s && s->type == JSON_BOOL) ? s->boolean : 0;
    if (!result && s && s->type == JSON_NUMBER) result = (int)s->number != 0;
    json_free(root);
    return result;
}

/* ============================================================
 * 生成核心 — 支持 SSE 流式输出
 *
 * stream_fd > 0 时: 流式模式, 每生成一个 token 就通过 SSE 推送
 * stream_fd <= 0 时: 非流式, 全部生成完写入 out 缓冲区
 * ============================================================ */
static int handle_completion(const char *prompt, float temperature,
                             int max_tokens, int top_k,
                             char *out, int out_max,
                             int stream_fd) {
    Config *cfg = &g_state.cfg;
    RunState *s = &g_state.s;
    Tokenizer *tk = &g_state.tk;

    /* 清空 KV Cache */
    int kvd = cfg->n_kv_heads * cfg->head_dim;
    memset(s->key_cache, 0,
           (size_t)cfg->n_layers * cfg->seq_len * kvd * sizeof(float));
    memset(s->value_cache, 0,
           (size_t)cfg->n_layers * cfg->seq_len * kvd * sizeof(float));

    /* 编码 prompt */
    int prompt_tokens[1024];
    int n_prompt = tokenizer_encode(tk, prompt, -1, prompt_tokens, 1024);
    if (n_prompt == 0) { out[0] = 0; return 0; }

    int total_len = n_prompt + max_tokens;
    if (total_len > cfg->seq_len) total_len = cfg->seq_len;

    unsigned int rng_state = 12345;
    int out_pos = 0;
    int token = prompt_tokens[0];
    int n_generated = 0;

    /* 如果是流式模式, 先发 SSE 头 */
    if (stream_fd > 0) {
        send_stream_headers(stream_fd);
    }

    for (int pos = 0; pos < total_len; pos++) {
        forward(cfg, &g_state.w, s, token, pos);

        int next;
        if (pos < n_prompt - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            if (temperature == 0.0f) {
                next = 0;
                float bestv = s->logits[0];
                for (int i = 1; i < cfg->vocab_size; i++) {
                    if (s->logits[i] > bestv) { bestv = s->logits[i]; next = i; }
                }
            } else {
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

        /* 检测结束条件: EOS 或 <|im_end|> (151645) */
        if (pos >= n_prompt - 1) {
            if (next == tk->eos_id) break;
            if (next == 151644 || next == 151645) break; /* <|im_start|>/<|im_end|> */
        }

        if (pos >= n_prompt - 1) {
            char buf[256];
            int blen = tokenizer_decode(tk, next, buf, sizeof(buf));

            if (stream_fd > 0) {
                /* ★ 流式模式: 每个 token 立即通过 SSE 推送 */
                if (blen > 0) {
                    char escaped[512];
                    json_escape(buf, escaped, sizeof(escaped));
                    char sse[1024];
                    int sse_len = snprintf(sse, sizeof(sse),
                        "data: {\"choices\":[{\"delta\":{\"content\":\"%s\"}}]}\n\n",
                        escaped);
                    write(stream_fd, sse, sse_len);
                }
            } else {
                /* 非流式: 写入缓冲区 */
                if (out_pos + blen < out_max - 1) {
                    memcpy(out + out_pos, buf, blen);
                    out_pos += blen;
                }
            }
            n_generated++;
        }
        token = next;
    }

    if (stream_fd > 0) {
        /* 流式结束: 发送 [DONE] */
        const char *done = "data: [DONE]\n\n";
        write(stream_fd, done, strlen(done));
    } else {
        out[out_pos] = 0;
    }
    return n_generated;
}

/* ============================================================
 * 处理单个连接 (在子进程里调用)
 * ============================================================ */
static void handle_connection(int client_fd, const char *client_ip,
                              float default_temp, int default_top_k) {
    char path[256];
    char body[65536];
    int body_len = read_http_request(client_fd, path, sizeof(path),
                                     body, sizeof(body));
    if (body_len < 0) {
        close(client_fd);
        return;
    }

    char response[65536];

    if (strcmp(path, "/v1/models") == 0) {
        build_models_response(response, sizeof(response));
        send_response(client_fd, 200, "OK", "application/json", response);
        fprintf(stderr, "[server] %s GET /v1/models → 200\n", client_ip);

    } else if (strcmp(path, "/v1/chat/completions") == 0 && body_len > 0) {
        char *prompt = extract_chat_prompt(body);
        if (!prompt) {
            build_error_response(response, sizeof(response),
                "messages field is required");
            send_response(client_fd, 400, "Bad Request", "application/json", response);
        } else {
            float temp = extract_temperature(body, default_temp);
            int max_tok = extract_max_tokens(body, 200);
            int stream = extract_stream(body);

            fprintf(stderr, "[server] %s chat completions stream=%d temp=%.1f ...\n",
                    client_ip, stream, temp);

            if (stream) {
                /* ★ 流式: 边生成边推送 */
                handle_completion(prompt, temp, max_tok, default_top_k,
                                  NULL, 0, client_fd);
                fprintf(stderr, "[server] %s → 200 (streamed)\n", client_ip);
            } else {
                /* 非流式: 全部生成完再返回 */
                char generated[32768];
                int n_generated = handle_completion(prompt, temp, max_tok,
                                                     default_top_k, generated,
                                                     sizeof(generated), -1);
                build_chat_response(response, sizeof(response),
                                    generated, 0, n_generated);
                send_response(client_fd, 200, "OK", "application/json", response);
                fprintf(stderr, "[server] %s → 200 (生成 %d token)\n",
                        client_ip, n_generated);
            }
            free(prompt);
        }

    } else {
        build_error_response(response, sizeof(response), "not found");
        send_response(client_fd, 404, "Not Found", "application/json", response);
        fprintf(stderr, "[server] %s %s → 404\n", client_ip, path);
    }

    close(client_fd);
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
    fprintf(stderr, "[server]   POST /v1/chat/completions  (OpenAI 兼容, 支持流式+多轮)\n");
    fprintf(stderr, "[server]   GET  /v1/models\n");
    fprintf(stderr, "[server] 按 Ctrl+C 退出\n\n");

    /* 忽略 SIGPIPE + 自动回收子进程 (防僵尸) */
    signal(SIGPIPE, SIG_IGN);
    signal(SIGCHLD, SIG_IGN);  /* ★ 任务③: 自动回收 fork 的子进程 */

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

        /* ★ 任务③: fork() 实现并发
         * 父进程: 立即 close(client_fd), 回到 accept
         * 子进程: 继承已加载的模型 (copy-on-write), 处理请求后 exit */
        pid_t pid = fork();
        if (pid == 0) {
            /* 子进程: 处理这个连接 */
            close(listen_fd);  /* 子进程不需要监听 socket */
            handle_connection(client_fd, client_ip, default_temp, default_top_k);
            exit(0);  /* 处理完直接退出 */
        } else if (pid > 0) {
            /* 父进程: 关闭 client_fd (子进程会用自己的副本) */
            close(client_fd);
        } else {
            perror("fork");
            close(client_fd);
        }
    }

    close(listen_fd);
}
