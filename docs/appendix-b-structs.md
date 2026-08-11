# 附录 B：结构体速查

> [上一章：第 10 章 生态对比](./10-ecosystem.md) | [上一附录：附录 A 术语总表](./appendix-a-glossary.md)

本项目所有源文件里的结构体，集中放在这里方便速查。

---

## B.1 Config — 模型配置（`net.h`）

所有维度参数，来自 `config.json`：

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

## B.2 TransformerWeights — 所有权重（`net.h`）

```c
typedef struct {
    float *token_embedding_table;  // (151936, 896) 查表: token→向量
    float *rms_att_weight;         // (24, 896)     attention 前的归一化
    float *rms_ffn_weight;         // (24, 896)     MLP 前的归一化
    float *wq, *wk, *wv, *wo;      // 注意力的 Q/K/V/O 投影权重
    float *bq, *bk, *bv;           // Q/K/V 偏置 (Qwen 独有, O 无偏置)
    float *w_gate, *w_up, *w_down; // SwiGLU 的三个矩阵
    float *rms_final_weight;       // (896,) 最终归一化
} TransformerWeights;
```

## B.3 RunState — 工作内存（`net.h`）

```c
typedef struct {
    float *x, *xb, *xb2;    // 临时向量
    float *hb, *hb2;        // MLP 临时向量
    float *q, *k, *v;       // 当前 token 的 Q/K/V
    float *att;             // 注意力分数
    float *logits;          // 最终输出
    float *key_cache;       // ★ KV Cache (跨步累积)
    float *value_cache;
} RunState;
```

## B.4 SafetensorsFile — 文件句柄（`safetensors.h`）

```c
typedef struct {
    char *path;            // 文件路径 (调试用)
    char *header_json;     // header JSON 文本
    unsigned long header_len;
    JsonValue *header;     // 解析后的 JSON 树
    unsigned char *data;   // 指向 raw buffer 起点
    unsigned long data_size;
    unsigned char *mapped; // mmap 起点
    unsigned long mapped_size;
} SafetensorsFile;
```

## B.5 JsonValue — JSON 树节点（`json.h`）

```c
typedef struct JsonValue {
    JsonType type;           // NULL/BOOL/NUMBER/STRING/ARRAY/OBJECT
    double number;           // type==NUMBER 时的值
    int boolean;             // type==BOOL 时的值
    char *string;            // type==STRING 时的值
    struct JsonValue *first_child;   // 第一个子节点
    struct JsonValue *next_sibling;  // 下一个兄弟
    struct JsonValue *value_child;   // OBJECT 的 key 指向 value
} JsonValue;
```

## B.6 Tokenizer — 分词器（`tokenizer.h`）

```c
typedef struct {
    int vocab_size;       // 词表大小
    char **vocab;         // vocab[i] = token i 的字符串
    int *vocab_len;       // vocab[i] 的字节长度
    int n_merges;         // BPE 合并规则数
    int *merges;          // merges[2i]=a, merges[2i+1]=b
    int hash_size;        // 哈希桶数
    int *hash_buckets;    // 桶头
    int *hash_next;       // 冲突链表
    int n_special;        // special token 数
    char byte_map_chars[256][8];  // byte→unicode 映射
    int bos_id, eos_id, pad_id;
} Tokenizer;
```

## B.7 TraceRecord / TraceLog — 日志（`trace.h`）

```c
typedef struct {
    LogLevel    level;    // 日志等级
    LogCategory cat;      // 日志类别 (决定颜色)
    int         indent;   // 缩进层级
    char       *text;     // 已格式化的文本
} TraceRecord;

typedef struct {
    TraceRecord *records;
    int          count;
    int          capacity;
} TraceLog;
```

## B.8 RNG — 随机数生成器（`run.c`）

```c
typedef struct { unsigned int x, y, z, w; } RNG;  // xorshift128 状态
```
