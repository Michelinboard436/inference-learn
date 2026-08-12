#ifndef SERVER_H
#define SERVER_H

/*
 * server.h — 纯 C HTTP 服务器 (OpenAI 兼容 API)
 *
 * 启动后常驻: 加载模型一次, 之后通过 HTTP 接收请求。
 *
 * 用法:
 *   ./run serve model.safetensors tokenizer.bin [--port 8000]
 *
 * 端点:
 *   POST /v1/chat/completions   — 聊天补全 (OpenAI 兼容)
 *   GET  /v1/models             — 返回模型列表
 *
 * 请求格式 (OpenAI 标准):
 *   {"messages":[{"role":"user","content":"Hello"}],"temperature":0.8}
 *
 * 响应格式 (OpenAI 标准):
 *   {"choices":[{"message":{"content":"Hi!"}}],"usage":{...}}
 */

/* 启动 HTTP 服务器。阻塞调用, 不会返回。 */
void run_server(const char *model_path, const char *tok_path,
                int port, float default_temp, int default_top_k);

#endif // SERVER_H
