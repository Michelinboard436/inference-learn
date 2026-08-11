/* safetensors.c — 读取 safetensors 文件
 *
 * 二进制布局回顾:
 *   [u64 header_len] [header_len 字节 JSON] [raw tensor data...]
 *
 * 我们 mmap 整个文件，零拷贝地拿到 raw buffer 指针。
 * 加载权重时把 bf16 转成 fp32 (bf16 = fp32 的高 16 位)。
 *
 * bf16 -> fp32 的位技巧:
 *   bf16 是 1 位符号 + 8 位指数 + 7 位尾数，正好是 fp32 的高 16 位。
 *   所以只要把 bf16 的 16 bit 放到 fp32 的最高 16 位，低 16 位填 0:
 *     uint32_t fp32_bits = ((uint32_t)bf16) << 16;
 *     float f;
 *     memcpy(&f, &fp32_bits, 4);   // 不能直接 *(float*)& 解引用，违反严格别名
 */

#include "safetensors.h"
#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* mmap 在 macOS/Linux 上接口一致，都在 sys/mman.h */
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* 读小端 u64 */
static unsigned long read_u64_le(const unsigned char *b) {
    unsigned long v = 0;
    for (int i = 0; i < 8; i++) {
        v |= ((unsigned long)b[i]) << (8 * i);
    }
    return v;
}

int safetensors_open(const char *path, SafetensorsFile *st) {
    memset(st, 0, sizeof(*st));
    st->path = strdup(path);

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return -1; }

    struct stat sb;
    if (fstat(fd, &sb) < 0) { perror("fstat"); close(fd); return -1; }
    unsigned long file_size = (unsigned long)sb.st_size;

    void *map = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) { perror("mmap"); return -1; }

    if (file_size < 8) {
        fprintf(stderr, "safetensors: 文件太小\n");
        munmap(map, file_size); return -1;
    }

    unsigned char *bytes = (unsigned char *)map;
    unsigned long header_len = read_u64_le(bytes);
    if (8 + header_len > file_size) {
        fprintf(stderr, "safetensors: header 长度 %lu 超过文件大小 %lu\n",
                header_len, file_size);
        munmap(map, file_size); return -1;
    }

    /* 拷贝一份 header JSON 文本，加 '\0' */
    char *hdr = (char *)malloc(header_len + 1);
    memcpy(hdr, bytes + 8, header_len);
    hdr[header_len] = 0;

    st->header_json = hdr;
    st->header_len  = header_len;
    st->mapped      = bytes;
    st->mapped_size = file_size;
    st->data        = bytes + 8 + header_len;
    st->data_size   = file_size - (8 + header_len);

    /* 解析一次 header JSON，缓存到 st->header，后续 find 直接用 */
    st->header = json_parse(hdr);
    if (!st->header) {
        fprintf(stderr, "safetensors: header 不是合法 JSON\n");
        munmap(st->mapped, st->mapped_size);
        free(st->header_json); free(st->path);
        memset(st, 0, sizeof(*st));
        return -1;
    }
    return 0;
}

void safetensors_close(SafetensorsFile *st) {
    if (st->header) json_free(st->header);
    if (st->mapped) munmap(st->mapped, st->mapped_size);
    if (st->header_json) free(st->header_json);
    if (st->path) free(st->path);
    memset(st, 0, sizeof(*st));
}

int safetensors_find(SafetensorsFile *st, const char *name,
                     unsigned long *out_offset, unsigned long *out_nbytes,
                     int *out_shape, int *out_ndim, const char **out_dtype) {
    JsonValue *root = st->header;
    if (!root) {
        fprintf(stderr, "safetensors: header 未解析\n");
        return -1;
    }
    JsonValue *t = json_object_get(root, name);
    if (!t) return -1;

    /* dtype / shape / data_offsets */
    JsonValue *dtype = json_object_get(t, "dtype");
    JsonValue *shape = json_object_get(t, "shape");
    JsonValue *doff  = json_object_get(t, "data_offsets");

    if (!dtype || !shape || !doff) {
        fprintf(stderr, "safetensors: 张量 %s 缺字段\n", name);
        return -1;
    }

    /* dtype 字符串 */
    const char *dt = (dtype->type == JSON_STRING) ? dtype->string : "";
    if (out_dtype) *out_dtype = dt;

    /* shape */
    int ndim = json_array_len(shape);
    if (out_ndim) *out_ndim = ndim;
    if (out_shape) {
        for (int i = 0; i < ndim && i < 4; i++) {
            out_shape[i] = (int)json_array_get(shape, i)->number;
        }
    }

    /* data_offsets: [start, end] */
    unsigned long off0 = (unsigned long)json_array_get(doff, 0)->number;
    unsigned long off1 = (unsigned long)json_array_get(doff, 1)->number;
    if (out_offset) *out_offset = off0;
    if (out_nbytes)  *out_nbytes  = off1 - off0;

    return 0;
}

/* 把单个 bf16/fp32 张量读成 float 数组 */
float *safetensors_load_float(SafetensorsFile *st, const char *name) {
    unsigned long off, nbytes;
    int shape[4], ndim;
    const char *dtype = NULL;
    if (safetensors_find(st, name, &off, &nbytes, shape, &ndim, &dtype) < 0) {
        fprintf(stderr, "safetensors: 找不到张量 '%s'\n", name);
        return NULL;
    }

    const unsigned char *src = st->data + off;

    if (strcmp(dtype, "F32") == 0) {
        /* 直接拷贝 (小端架构) */
        float *out = (float *)malloc(nbytes);
        if (!out) return NULL;
        /* 注意: 多数 PC/macOS 都是小端，可以直接 memcpy。
           严格起见应该按字节重组，但这里假设小端架构。 */
        memcpy(out, src, nbytes);
        return out;
    }

    if (strcmp(dtype, "BF16") == 0) {
        size_t n = nbytes / 2;   /* bf16 每元素 2 字节 */
        float *out = (float *)malloc(n * sizeof(float));
        if (!out) return NULL;
        for (size_t i = 0; i < n; i++) {
            uint16_t b = (uint16_t)(src[2*i] | (src[2*i+1] << 8)); /* 小端读 bf16 */
            uint32_t bits = ((uint32_t)b) << 16;   /* 放到 fp32 高 16 位 */
            memcpy(&out[i], &bits, 4);
        }
        return out;
    }

    if (strcmp(dtype, "F16") == 0) {
        /* fp16 -> fp32，偶尔会用到 */
        size_t n = nbytes / 2;
        float *out = (float *)malloc(n * sizeof(float));
        if (!out) return NULL;
        for (size_t i = 0; i < n; i++) {
            uint16_t h = (uint16_t)(src[2*i] | (src[2*i+1] << 8));
            /* fp16 -> fp32 标准转换 */
            uint32_t sign = (h & 0x8000) << 16;
            uint32_t exp  = (h & 0x7C00) >> 10;
            uint32_t mant = (h & 0x03FF);
            uint32_t bits;
            if (exp == 0) {
                if (mant == 0) {
                    bits = sign;   /* +-0 */
                } else {
                    /* 非正规数，重新对齐 */
                    int e = -1;
                    do { e++; mant <<= 1; } while ((mant & 0x0400) == 0);
                    mant &= 0x03FF;
                    bits = sign | ((127 - 15 - e) << 23) | (mant << 13);
                }
            } else if (exp == 0x1F) {
                bits = sign | 0x7F800000 | (mant << 13);  /* inf/nan */
            } else {
                bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
            }
            memcpy(&out[i], &bits, 4);
        }
        return out;
    }

    fprintf(stderr, "safetensors: 不支持的 dtype '%s' (张量 %s)\n", dtype, name);
    return NULL;
}

/* 按层叠放权重的帮助宏: 逐层从 safetensors 里读出 "layers.{l}.xxx" 并拼到一起 */
#define LAYER_NAME(buf, sizeof_buf, l, suffix) \
    snprintf((buf), (sizeof_buf), "model.layers.%d.%s", (l), (suffix))

/* 逐层读取并拼成连续的 (n_layers, ...) 数组。
 * each_floats: 每层该张量的元素个数 */
static int load_layered(SafetensorsFile *st, float *dst,
                        const char *suffix, int n_layers, size_t each_floats) {
    char name[256];
    for (int l = 0; l < n_layers; l++) {
        LAYER_NAME(name, sizeof(name), l, suffix);
        float *tmp = safetensors_load_float(st, name);
        if (!tmp) {
            fprintf(stderr, "加载层 %d 的 %s 失败\n", l, suffix);
            return -1;
        }
        memcpy(dst + (size_t)l * each_floats, tmp, each_floats * sizeof(float));
        free(tmp);
    }
    return 0;
}

int safetensors_load_weights(SafetensorsFile *st, Config *config,
                             TransformerWeights *w) {
    int L = config->n_layers;
    int dim = config->dim;
    int hd  = config->hidden_dim;
    int qd  = config->n_heads    * config->head_dim;   /* = dim */
    int kvd = config->n_kv_heads * config->head_dim;   /* = 128 */

    /* 分配 */
    w->token_embedding_table = safetensors_load_float(st, "model.embed_tokens.weight");
    if (!w->token_embedding_table) return -1;

    /* Qwen 把各层权重命名为:
     *   model.layers.{l}.input_layernorm.weight        -> rms_att_weight
     *   model.layers.{l}.self_attn.q_proj.weight/bias
     *   model.layers.{l}.self_attn.k_proj.weight/bias
     *   model.layers.{l}.self_attn.v_proj.weight/bias
     *   model.layers.{l}.self_attn.o_proj.weight       (无 bias)
     *   model.layers.{l}.post_attention_layernorm.weight -> rms_ffn_weight
     *   model.layers.{l}.mlp.gate_proj.weight
     *   model.layers.{l}.mlp.up_proj.weight
     *   model.layers.{l}.mlp.down_proj.weight
     */
    w->rms_att_weight = malloc((size_t)L * dim * sizeof(float));
    w->rms_ffn_weight = malloc((size_t)L * dim * sizeof(float));
    w->wq = malloc((size_t)L * qd  * dim * sizeof(float));
    w->wk = malloc((size_t)L * kvd * dim * sizeof(float));
    w->wv = malloc((size_t)L * kvd * dim * sizeof(float));
    w->wo = malloc((size_t)L * qd  * dim * sizeof(float));
    w->bq = malloc((size_t)L * qd       * sizeof(float));
    w->bk = malloc((size_t)L * kvd      * sizeof(float));
    w->bv = malloc((size_t)L * kvd      * sizeof(float));
    w->w_gate = malloc((size_t)L * hd * dim * sizeof(float));
    w->w_up   = malloc((size_t)L * hd * dim * sizeof(float));
    w->w_down = malloc((size_t)L * dim * hd * sizeof(float));
    if (!w->rms_att_weight || !w->rms_ffn_weight ||
        !w->wq || !w->wk || !w->wv || !w->wo ||
        !w->bq || !w->bk || !w->bv ||
        !w->w_gate || !w->w_up || !w->w_down) {
        fprintf(stderr, "权重分配失败\n"); return -1;
    }

    /* 逐层拼装 */
    if (load_layered(st, w->rms_att_weight, "input_layernorm.weight",        L, dim) != 0) return -1;
    if (load_layered(st, w->rms_ffn_weight, "post_attention_layernorm.weight", L, dim) != 0) return -1;

    if (load_layered(st, w->wq, "self_attn.q_proj.weight", L, (size_t)qd  * dim) != 0) return -1;
    if (load_layered(st, w->wk, "self_attn.k_proj.weight", L, (size_t)kvd * dim) != 0) return -1;
    if (load_layered(st, w->wv, "self_attn.v_proj.weight", L, (size_t)kvd * dim) != 0) return -1;
    if (load_layered(st, w->wo, "self_attn.o_proj.weight", L, (size_t)qd  * dim) != 0) return -1;

    /* ★ Qwen 独有: Q/K/V 有 bias */
    if (load_layered(st, w->bq, "self_attn.q_proj.bias", L, qd)  != 0) return -1;
    if (load_layered(st, w->bk, "self_attn.k_proj.bias", L, kvd) != 0) return -1;
    if (load_layered(st, w->bv, "self_attn.v_proj.bias", L, kvd) != 0) return -1;

    if (load_layered(st, w->w_gate, "mlp.gate_proj.weight", L, (size_t)hd * dim) != 0) return -1;
    if (load_layered(st, w->w_up,   "mlp.up_proj.weight",   L, (size_t)hd * dim) != 0) return -1;
    if (load_layered(st, w->w_down, "mlp.down_proj.weight", L, (size_t)dim * hd) != 0) return -1;

    /* final norm: model.norm.weight */
    w->rms_final_weight = safetensors_load_float(st, "model.norm.weight");
    if (!w->rms_final_weight) return -1;

    /* tie_word_embeddings=true: 不读 lm_head.weight，直接复用 token_embedding */
    /* (token_embedding_table 既是 embedding 也是 lm_head) */

    fprintf(stderr, "[加载] 完整权重已载入内存 (fp32)\n");
    return 0;
}
