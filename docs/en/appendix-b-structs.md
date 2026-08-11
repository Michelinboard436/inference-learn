# Appendix B · Struct Reference

> 🌐 [中文版](../cn/appendix-b-structs) | English

> [Previous: Chapter 10 · Ecosystem Comparison](./10-ecosystem.md) | [Previous appendix: Appendix A · Glossary](./appendix-a-glossary.md)

All the structs across this project's source files, gathered here for quick reference.

---

## B.1 Config — Model Configuration (`net.h`)

All dimension parameters, sourced from `config.json`:

```c
typedef struct {
    int dim;          // 896    residual-stream width (token-vector length)
    int hidden_dim;   // 4864   MLP intermediate width
    int n_layers;     // 24     number of Transformer layers
    int n_heads;      // 14     number of Query heads
    int n_kv_heads;   // 2      number of KV heads (GQA, smaller than n_heads)
    int head_dim;     // 64     dimension of each head
    int vocab_size;   // 151936 vocabulary size
    int seq_len;      // 2048   maximum sequence length
    float rope_theta; // 1e6    RoPE base frequency
    float rms_eps;    // 1e-6   RMSNorm zero-division guard
} Config;
```

## B.2 TransformerWeights — All Weights (`net.h`)

```c
typedef struct {
    float *token_embedding_table;  // (151936, 896) lookup: token → vector
    float *rms_att_weight;         // (24, 896)     normalization before attention
    float *rms_ffn_weight;         // (24, 896)     normalization before MLP
    float *wq, *wk, *wv, *wo;      // attention Q/K/V/O projection weights
    float *bq, *bk, *bv;           // Q/K/V biases (Qwen-specific; O has no bias)
    float *w_gate, *w_up, *w_down; // SwiGLU's three matrices
    float *rms_final_weight;       // (896,) final normalization
} TransformerWeights;
```

## B.3 RunState — Working Memory (`net.h`)

```c
typedef struct {
    float *x, *xb, *xb2;    // scratch vectors
    float *hb, *hb2;        // MLP scratch vectors
    float *q, *k, *v;       // Q/K/V of the current token
    float *att;             // attention scores
    float *logits;          // final output
    float *key_cache;       // ★ KV Cache (accumulated across steps)
    float *value_cache;
} RunState;
```

## B.4 SafetensorsFile — File Handle (`safetensors.h`)

```c
typedef struct {
    char *path;            // file path (for debugging)
    char *header_json;     // header JSON text
    unsigned long header_len;
    JsonValue *header;     // parsed JSON tree
    unsigned char *data;   // points to the start of the raw buffer
    unsigned long data_size;
    unsigned char *mapped; // mmap start
    unsigned long mapped_size;
} SafetensorsFile;
```

## B.5 JsonValue — JSON Tree Node (`json.h`)

```c
typedef struct JsonValue {
    JsonType type;           // NULL/BOOL/NUMBER/STRING/ARRAY/OBJECT
    double number;           // value when type==NUMBER
    int boolean;             // value when type==BOOL
    char *string;            // value when type==STRING
    struct JsonValue *first_child;   // first child node
    struct JsonValue *next_sibling;  // next sibling
    struct JsonValue *value_child;   // for OBJECT, the key points to its value
} JsonValue;
```

## B.6 Tokenizer — Tokenizer (`tokenizer.h`)

```c
typedef struct {
    int vocab_size;       // vocabulary size
    char **vocab;         // vocab[i] = the string of token i
    int *vocab_len;       // byte length of vocab[i]
    int n_merges;         // number of BPE merge rules
    int *merges;          // merges[2i]=a, merges[2i+1]=b
    int hash_size;        // number of hash buckets
    int *hash_buckets;    // bucket heads
    int *hash_next;       // collision chain
    int n_special;        // number of special tokens
    char byte_map_chars[256][8];  // byte → unicode mapping
    int bos_id, eos_id, pad_id;
} Tokenizer;
```

## B.7 TraceRecord / TraceLog — Logging (`trace.h`)

```c
typedef struct {
    LogLevel    level;    // log level
    LogCategory cat;      // log category (determines color)
    int         indent;   // indentation level
    char       *text;     // already-formatted text
} TraceRecord;

typedef struct {
    TraceRecord *records;
    int          count;
    int          capacity;
} TraceLog;
```

## B.8 RNG — Random Number Generator (`run.c`)

```c
typedef struct { unsigned int x, y, z, w; } RNG;  // xorshift128 state
```
