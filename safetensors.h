#ifndef SAFETENSORS_H
#define SAFETENSORS_H

/*
 * safetensors.h — 读取 safetensors 文件
 *
 * safetensors 二进制布局 (很简单，这也是它流行的原因):
 *   [ 8 字节 little-endian uint64 = header JSON 的字节长度 N ]
 *   [ N 字节 UTF-8 JSON header ]
 *   [ 剩余全部 = 连续的 raw tensor 数据 buffer ]
 *
 * header 是一个 JSON object，例如:
 *   {
 *     "model.embed_tokens.weight": {
 *       "dtype": "BF16",
 *       "shape": [151936, 896],
 *       "data_offsets": [0, 271648768]
 *     },
 *     ...
 *     "__metadata__": { ... }
 *   }
 *
 * data_offsets 是相对于 raw buffer 起点的字节偏移。
 */

#include "net.h"
#include "json.h"

/* 加载后的 safetensors 文件句柄 */
typedef struct {
    char *path;            // 文件路径 (调试用)
    char *header_json;     // header JSON 文本 (以 \0 结尾)
    unsigned long header_len; // header JSON 的字节长度
    JsonValue *header;     // 解析后的 header 树 (open 时建好，close 时释放)
    unsigned char *data;   // 指向 raw buffer 起点的指针
    unsigned long data_size; // raw buffer 字节数
    /* 整个文件 mmap 的起点 (用来最后 munmap) */
    unsigned char *mapped;
    unsigned long mapped_size;
} SafetensorsFile;

/* 打开并解析 safetensors 文件。成功返回 0。 */
int safetensors_open(const char *path, SafetensorsFile *st);

/* 关闭、释放资源 */
void safetensors_close(SafetensorsFile *st);

/* 查找一个张量，返回它的字节偏移和长度(原始 dtype)。
 * 找不到返回 -1。shape 通过 out_shape 返回(最多 4 维)。
 * 返回的 dtype 字符串("BF16"/"F32"/...)指针指向 header，无需释放。
 */
int safetensors_find(SafetensorsFile *st, const char *name,
                     unsigned long *out_offset, unsigned long *out_nbytes,
                     int *out_shape, int *out_ndim, const char **out_dtype);

/* 把一个 bf16/fp32 张量加载到 float* (自动转 fp32)。
 * - 从 st 的 raw buffer 读取
 * - bf16: 每个 16-bit 元素扩成 fp32
 * - fp32: 直接 memcpy
 * 返回 malloc 出来的 float 数组(用完 free)，或 NULL 失败。
 */
float *safetensors_load_float(SafetensorsFile *st, const char *name);

/* 把权重按 net.h 的布局加载到 TransformerWeights。
 * config 必须已经填好维度。
 * 返回 0 成功。
 */
int safetensors_load_weights(SafetensorsFile *st, Config *config,
                             TransformerWeights *w);

#endif // SAFETENSORS_H
