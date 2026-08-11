#ifndef TOKENIZER_H
#define TOKENIZER_H

/*
 * tokenizer.h — BPE 分词器
 *
 * 工作流:
 *   export_tokenizer.py 把 HF 的 tokenizer.json 转成 tokenizer.bin (紧凑二进制)
 *   本文件解析 tokenizer.bin，在 C 里实现 BPE 编码 + 解码。
 *
 * tokenizer.bin 布局 (全部小端):
 *   uint32  vocab_size           // 词表条目数 N
 *   uint32  n_merges             // BPE 合并规则数 M
 *   uint32  n_special            // special token 数
 *   后跟 N 条 vocab 字符串: 每条 = uint32 len + len 字节(不含\0)
 *   后跟 M 条 merge: 每条 = uint32 a + uint32 b  (表示 token[a]+token[b] -> 新 token)
 *   后跟 n_special 条 special: 每条 = uint32 id + uint32 len + len 字节
 *   后跟 256 字节 byte_unicode_map: 把 raw byte(0..255) 映射到 unicode char 的索引
 *
 * BPE 编码算法(和 GPT-2/Qwen 一致):
 *   1. 文本 → UTF-8 字节
 *   2. 每个 byte 经 byte-level 映射变成一个 unicode "字符"
 *   3. 用 pre-tokenizer 正则把文本切成 pre-tokens (空格/标点/字母数字)
 *   4. 对每个 pre-token: 初始每个"字符"是单独 token，贪心找最高优先级的 merge
 *   5. 重复直到无法 merge
 */

#include <stdio.h>

typedef struct {
    int   vocab_size;
    char **vocab;        // vocab[i] = token i 对应的 UTF-8 字符串(byte-level 编码)
    int  *vocab_len;     // vocab[i] 的字节长度
    int   n_merges;
    int  *merges;        // 长度 2*n_merges, merges[2i]=a, merges[2i+1]=b
                         // (a,b) 出现的顺序即优先级，越早优先级越高

    /* string -> id 查找表。用拉链法哈希。 */
    int    hash_size;     // 桶数 (2 的幂)
    int   *hash_buckets;  // 长度 hash_size, 存第一个 token id 或 -1
    int   *hash_next;     // 长度 vocab_size, 冲突时链到下一个 token id 或 -1

    /* special tokens (id -> string) */
    int   n_special;
    int  *special_ids;
    char **special_strs;
    int  *special_lens;

    /* byte_map: 第 i 个字节映射成什么 unicode 字符串(GPT-2 byte-level) */
    char  byte_map_chars[256][8];
    int   byte_map_len[256];

    /* bos / eos / pad 等 id */
    int   bos_id;
    int   eos_id;
    int   pad_id;
} Tokenizer;

/* 从 tokenizer.bin 加载。失败返回非 0。 */
int tokenizer_load(Tokenizer *t, const char *path);

void tokenizer_free(Tokenizer *t);

/* 把文本编码成 token id 序列。
 * - max_tokens: 输出数组容量
 * - 返回写入的 token 数；bos 若 !=-1 会前置
 */
int tokenizer_encode(Tokenizer *t, const char *text, int bos_id,
                     int *out_tokens, int max_tokens);

/* 把单个 token id 解码成 UTF-8 字节，追加到 buf。
 * - 返回写入的字节数；buf 剩余空间不足会截断。
 * - next_id: 下一个 token (Qwen 用它决定当前空格/标点的渲染，简化版可忽略)
 */
int tokenizer_decode(Tokenizer *t, int id, char *buf, int buf_cap);

#endif // TOKENIZER_H
