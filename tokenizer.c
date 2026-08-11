/* tokenizer.c — BPE 分词器 (GPT-2 ByteLevel 风格, Qwen 使用)
 *
 * 编码流程 (和 HuggingFace tokenizers 完全一致):
 *   1. 输入 UTF-8 文本
 *   2. 用 GPT-2 正则切成 pre-tokens (字母串/数字/空格+词/标点串/换行/空白)
 *   3. 每个 pre-token 的字节经 byte_map 映射成 unicode "字符"
 *      (例如空格 0x20 -> 'Ġ' U+0120, 这样 BPE 词表里没有控制字符)
 *   4. 映射后的"字符"序列初始时每个字符是一个 token id (查 vocab)
 *   5. 贪心 BPE: 反复找优先级最高(merge 序号最小)的相邻对合并, 直到无法合并
 *
 * 解码流程 (反向):
 *   token id -> vocab 字符串 (byte-level 编码) -> 还原回原始字节
 */

#include "tokenizer.h"
#include "trace.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============ 内部: 哈希表 (string -> id) ============ */

/* FNV-1a 哈希, 对字符串做散列 */
static unsigned int hash_str(const char *s, int len) {
    unsigned int h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= (unsigned char)s[i];
        h *= 16777619u;
    }
    return h;
}

/* 建 vocab 的 string->id 索引 */
static int build_vocab_index(Tokenizer *t) {
    /* 桶数取大于 vocab_size 的最小 2 的幂 ×2, 负载因子 ~0.5 */
    int sz = 1;
    while (sz < t->vocab_size * 2) sz <<= 1;
    t->hash_size = sz;
    t->hash_buckets = malloc(sz * sizeof(int));
    t->hash_next    = malloc(t->vocab_size * sizeof(int));
    if (!t->hash_buckets || !t->hash_next) return -1;
    for (int i = 0; i < sz; i++) t->hash_buckets[i] = -1;
    for (int i = 0; i < t->vocab_size; i++) t->hash_next[i] = -1;

    for (int id = 0; id < t->vocab_size; id++) {
        unsigned int h = hash_str(t->vocab[id], t->vocab_len[id]) & (sz - 1);
        /* 头插法链入 */
        t->hash_next[id] = t->hash_buckets[h];
        t->hash_buckets[h] = id;
    }
    return 0;
}

/* 查 string -> id, 找不到返回 -1 */
static int vocab_lookup(Tokenizer *t, const char *s, int len) {
    unsigned int h = hash_str(s, len) & (t->hash_size - 1);
    for (int id = t->hash_buckets[h]; id != -1; id = t->hash_next[id]) {
        if (t->vocab_len[id] == len &&
            memcmp(t->vocab[id], s, len) == 0) {
            return id;
        }
    }
    return -1;
}

/* 查 (a, b) 这个相邻对的 merge 优先级。
 * 返回序号(越小优先级越高), 不在表里返回 INT_MAX。 */
#include <limits.h>
static int merge_rank(Tokenizer *t, int a, int b) {
    /* 朴素线性查找。n_merges=15万, 每对都线性扫太慢。
       实际 encode 里这会被调用很多次。我们建一个 (a,b)->rank 的哈希。
       简化起见这里先线性, 后面可优化。 */
    /* ★ 性能: 用一个和 vocab 一样大的二级哈希会更好, 但作为学习版先这样。 */
    /* 优化: 由于 merges 是有序的, 且 a 通常较小, 可以用二分。
       但 merges 按出现顺序存, 不是按 (a,b) 排序的。所以线性。
       实际上 15万次 × 序列长度 在长 prompt 下会慢, 但够学习用。 */
    for (int i = 0; i < t->n_merges; i++) {
        if (t->merges[2*i] == a && t->merges[2*i+1] == b) return i;
    }
    return INT_MAX;
}

/* ============ 加载 / 释放 ============ */

int tokenizer_load(Tokenizer *t, const char *path) {
    memset(t, 0, sizeof(*t));
    FILE *f = fopen(path, "rb");
    if (!f) { perror("打开 tokenizer.bin"); return -1; }

    unsigned int header[4];
    if (fread(header, sizeof(unsigned int), 4, f) != 4) {
        fprintf(stderr, "tokenizer.bin header 读取失败\n"); fclose(f); return -1;
    }
    t->vocab_size = header[0];
    t->n_merges   = header[1];
    t->n_special  = header[2];
    int max_len   = header[3];

    /* vocab */
    t->vocab     = malloc(t->vocab_size * sizeof(char *));
    t->vocab_len = malloc(t->vocab_size * sizeof(int));
    char *buf = malloc(max_len + 16);
    if (!t->vocab || !t->vocab_len || !buf) goto oom;
    for (int i = 0; i < t->vocab_size; i++) {
        unsigned int len;
        if (fread(&len, sizeof(unsigned int), 1, f) != 1) goto read_err;
        if (fread(buf, 1, len, f) != len) goto read_err;
        char *s = malloc(len + 1);
        memcpy(s, buf, len);
        s[len] = 0;
        t->vocab[i] = s;
        t->vocab_len[i] = (int)len;
    }
    free(buf); buf = NULL;

    /* merges */
    t->merges = malloc((size_t)t->n_merges * 2 * sizeof(int));
    if (!t->merges) goto oom;
    for (int i = 0; i < t->n_merges; i++) {
        unsigned int pair[2];
        if (fread(pair, sizeof(unsigned int), 2, f) != 2) goto read_err;
        t->merges[2*i]   = (int)pair[0];
        t->merges[2*i+1] = (int)pair[1];
    }

    /* specials */
    if (t->n_special > 0) {
        t->special_ids   = malloc(t->n_special * sizeof(int));
        t->special_strs  = malloc(t->n_special * sizeof(char *));
        t->special_lens  = malloc(t->n_special * sizeof(int));
        if (!t->special_ids || !t->special_strs || !t->special_lens) goto oom;
        for (int i = 0; i < t->n_special; i++) {
            unsigned int sid, len;
            if (fread(&sid, sizeof(unsigned int), 1, f) != 1) goto read_err;
            if (fread(&len,  sizeof(unsigned int), 1, f) != 1) goto read_err;
            char *s = malloc(len + 1);
            if (fread(s, 1, len, f) != len) { free(s); goto read_err; }
            s[len] = 0;
            t->special_ids[i]  = (int)sid;
            t->special_strs[i] = s;
            t->special_lens[i] = (int)len;
        }
    }

    /* byte_map: 256 条 (len + bytes) */
    for (int i = 0; i < 256; i++) {
        unsigned int len;
        if (fread(&len, sizeof(unsigned int), 1, f) != 1) goto read_err;
        if (len > 7) { fprintf(stderr, "byte_map[%d] len=%u 太长\n", i, len); goto read_err; }
        if (fread(t->byte_map_chars[i], 1, len, f) != len) goto read_err;
        t->byte_map_len[i] = (int)len;
    }

    fclose(f);

    /* 默认 special id */
    t->bos_id = -1; t->eos_id = -1; t->pad_id = -1;
    for (int i = 0; i < t->n_special; i++) {
        if (strcmp(t->special_strs[i], "<|endoftext|>") == 0) {
            t->eos_id = t->special_ids[i];
            t->bos_id = t->special_ids[i];
            t->pad_id = t->special_ids[i];
        }
    }

    /* 建 vocab 反向索引 */
    if (build_vocab_index(t) != 0) goto oom;

    return 0;

read_err:
    fprintf(stderr, "tokenizer.bin 读取中途失败\n");
oom:
    if (buf) free(buf);
    fclose(f);
    return -1;
}

void tokenizer_free(Tokenizer *t) {
    if (t->vocab) {
        for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
        free(t->vocab);
    }
    free(t->vocab_len);
    free(t->merges);
    free(t->hash_buckets);
    free(t->hash_next);
    if (t->special_strs) {
        for (int i = 0; i < t->n_special; i++) free(t->special_strs[i]);
        free(t->special_strs);
    }
    free(t->special_ids);
    free(t->special_lens);
    memset(t, 0, sizeof(*t));
}

/* ============ 编码 ============ */

/* 简化的 GPT-2 pre-tokenizer: 把文本切成 pre-token 的字节范围 [start,end)
 *
 * 返回 pre-token 个数, 范围写到 out_starts/out_ends (调用者分配)。
 * 近似 GPT-2 正则:
 *   (\\s)'?[A-Za-z]+  |  空格+可选撇号+字母串
 *   '?[A-Za-z]+       |  字母串
 *   [0-9]             |  单个数字
 *   [非字母数字非空白的 ASCII]+ | 标点串
 *   空白/换行串
 *   非 ASCII 字节序列(UTF-8 多字节字符, 如中文)每个字符独立
 *
 * 注意: GPT-2 会把空格"粘"到后面的词上 (一个 pre-token), 所以
 *       " hello" 是一个 pre-token, 编码后空格变成 'Ġ'。
 */
static int pretokenize(const char *text, int text_len,
                       int *out_starts, int *out_ends, int max_tokens) {
    int n = 0;
    int i = 0;
    /* GPT-2 正则近似:
     *   's|'t|'re|'ve|'m|'ll|'d     缩写
     *   [^ ]?[A-Za-z]+              可选前导非空格字符 + 字母串
     *   [0-9]                       单个数字
     *   其它非空白非字母数字 ASCII   标点串
     *   空白(空格/制表/换行)        连续空白自成一组
     *   非 ASCII (UTF-8)            每个完整字符一组
     *
     * 关键: GPT-2 会把"单个空格+字母"当一个 pre-token,
     *       这样 " world" 编码后空格变成 'Ġ' 并和 world 合并。
     */
    while (i < text_len && n < max_tokens) {
        unsigned char c = (unsigned char)text[i];

        /* 非 ASCII: 一个完整的 UTF-8 字符作为一个单元 */
        if (c >= 0x80) {
            int start = i;
            int clen = (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : 2;
            while (i < text_len && i < start + clen) i++;
            out_starts[n] = start;
            out_ends[n] = i;
            n++;
            continue;
        }

        /* 单个空格 + 后面是字母: 空格和字母串一起 (GPT-2 " ?\\p{L}+") */
        if (c == ' ' && i + 1 < text_len) {
            unsigned char nxt = (unsigned char)text[i+1];
            int is_letter = (nxt >= 'A' && nxt <= 'Z') || (nxt >= 'a' && nxt <= 'z');
            if (is_letter) {
                int start = i;   /* 包含前导空格 */
                i++;             /* 吃掉空格 */
                while (i < text_len) {
                    unsigned char d = (unsigned char)text[i];
                    if ((d >= 'A' && d <= 'Z') || (d >= 'a' && d <= 'z')) i++;
                    else break;
                }
                out_starts[n] = start;
                out_ends[n] = i;
                n++;
                continue;
            }
        }

        /* 空白 (空格/制表/换行): 连续空白自成一组
         * (走到这里说明不是"空格+字母"的情况) */
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            int start = i;
            while (i < text_len) {
                unsigned char d = (unsigned char)text[i];
                if (d == ' ' || d == '\t' || d == '\n' || d == '\r') i++;
                else break;
            }
            out_starts[n] = start;
            out_ends[n] = i;
            n++;
            continue;
        }

        /* 字母 (无前导空格): 连续字母 + 撇号缩写 */
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '\'') {
            int start = i;
            while (i < text_len) {
                unsigned char d = (unsigned char)text[i];
                if ((d >= 'A' && d <= 'Z') || (d >= 'a' && d <= 'z') || d == '\'') i++;
                else break;
            }
            out_starts[n] = start;
            out_ends[n] = i;
            n++;
            continue;
        }

        /* 数字: 单个数字作为一个 pre-token (GPT-2 \\p{N} 行为) */
        if (c >= '0' && c <= '9') {
            out_starts[n] = i;
            out_ends[n] = i + 1;
            i++;
            n++;
            continue;
        }

        /* 其它 ASCII 可打印字符: 标点串 */
        {
            int start = i;
            while (i < text_len) {
                unsigned char d = (unsigned char)text[i];
                if (d >= 0x80) break;
                if (d == ' ' || d == '\t' || d == '\n' || d == '\r') break;
                if ((d >= 'A' && d <= 'Z') || (d >= 'a' && d <= 'z') || d == '\'') break;
                if (d >= '0' && d <= '9') break;
                i++;
            }
            if (i == start) i++;   /* 保证前进 */
            out_starts[n] = start;
            out_ends[n] = i;
            n++;
        }
    }
    return n;
}

/* 把一个 pre-token 的字节范围做 byte-level 映射 + BPE, 输出 token id 序列。
 * 返回追加到 out_tokens 的个数。 */
static int encode_pretoken(Tokenizer *t,
                           const char *text, int start, int end,
                           int *out_tokens, int out_cap, int out_used) {
    /* 1) 每个字节映射成 byte-level 字符串, 拼成"字符"序列。
       这里"字符"是变长的字节, 我们用一个动态缓冲记录每个字符的偏移和长度。 */
    /* 映射后的连续字节缓冲 */
    char mapped[1024];
    int  char_off[512];   /* 每个"字符"在 mapped 里的起始偏移 */
    int  char_len[512];   /* 每个"字符"的字节长度 */
    int  nchars = 0;
    int  mapped_len = 0;

    for (int k = start; k < end; k++) {
        unsigned char b = (unsigned char)text[k];
        int clen = t->byte_map_len[b];
        const char *cstr = t->byte_map_chars[b];
        if (nchars >= 512) break;
        char_off[nchars] = mapped_len;
        char_len[nchars] = clen;
        memcpy(mapped + mapped_len, cstr, clen);
        mapped_len += clen;
        nchars++;
    }

    /* 2) 每个"字符"查 vocab 得到初始 token id 序列 */
    int ids[512];
    int nids = 0;
    for (int k = 0; k < nchars && k < 512; k++) {
        int id = vocab_lookup(t, mapped + char_off[k], char_len[k]);
        if (id < 0) {
            /* byte-level 保证每个单字节映射后都在 vocab 里;
               若仍然找不到(理论不该发生), 跳过 */
            id = t->vocab_size > 0 ? 0 : 0;
        }
        ids[nids++] = id;
    }

    /* 3) BPE 合并: 反复找优先级最高的相邻对合并 */
    int merge_step = 0;
    while (nids >= 2) {
        int best_rank = INT_MAX;
        int best_idx = -1;
        for (int k = 0; k < nids - 1; k++) {
            int r = merge_rank(t, ids[k], ids[k+1]);
            if (r < best_rank) {
                best_rank = r;
                best_idx = k;
            }
        }
        if (best_idx < 0) break;  /* 没有可合并的对 */

        /* 合并 (best_idx, best_idx+1): 用拼接后的字符串查 vocab 得新 id */
        int la = t->vocab_len[ids[best_idx]];
        int lb = t->vocab_len[ids[best_idx+1]];
        char merged[64];
        if (la + lb > 63) break;
        memcpy(merged, t->vocab[ids[best_idx]], la);
        memcpy(merged + la, t->vocab[ids[best_idx+1]], lb);
        int new_id = vocab_lookup(t, merged, la + lb);
        if (new_id < 0) break;

        LOGT_L(CAT_TOKENIZER, 2, "BPE合并#%d: 合并位置%d 的 id=%d+id=%d → id=%d (优先级=%d)",
               merge_step, best_idx, ids[best_idx], ids[best_idx+1], new_id, best_rank);

        ids[best_idx] = new_id;
        for (int k = best_idx + 1; k < nids - 1; k++) ids[k] = ids[k+1];
        nids--;
        merge_step++;
    }

    /* 4) 追加到输出 */
    for (int k = 0; k < nids; k++) {
        if (out_used + k >= out_cap) break;
        out_tokens[out_used + k] = ids[k];
    }
    return nids;
}

int tokenizer_encode(Tokenizer *t, const char *text, int bos_id,
                     int *out_tokens, int max_tokens) {
    int n_out = 0;
    if (bos_id >= 0 && n_out < max_tokens) {
        out_tokens[n_out++] = bos_id;
    }

    int text_len = (int)strlen(text);

    /* pre-tokenize: 最多 text_len 个 pre-token */
    int *starts = malloc(text_len * sizeof(int) + 4);
    int *ends   = malloc(text_len * sizeof(int) + 4);
    if (!starts || !ends) {
        free(starts); free(ends); return n_out;
    }
    int npt = pretokenize(text, text_len, starts, ends, text_len);

    LOGI_L(CAT_TOKENIZER, 0, "BPE 编码: \"%s\" → %d 个 pre-token", text, npt);

    for (int p = 0; p < npt && n_out < max_tokens; p++) {
        /* 打印 pre-token 内容 */
        char pretok[256];
        int plen = ends[p] - starts[p];
        if (plen > 255) plen = 255;
        memcpy(pretok, text + starts[p], plen);
        pretok[plen] = 0;
        LOGD_L(CAT_TOKENIZER, 1, "pre-token[%d] = \"%s\" (字节 %d..%d)",
               p, pretok, starts[p], ends[p]);

        int n_before = n_out;
        int added = encode_pretoken(t, text, starts[p], ends[p],
                                    out_tokens, max_tokens, n_out);
        n_out += added;

        /* 打印这个 pre-token 编码出的 id 序列 */
        if (added > 0) {
            char idstr[256];
            int pos = 0;
            for (int k = 0; k < added && pos < 240; k++) {
                pos += snprintf(idstr + pos, sizeof(idstr) - pos, "%d%s",
                                out_tokens[n_before + k], k+1<added ? "," : "");
            }
            LOGD_L(CAT_TOKENIZER, 1, "  → %d 个 token: [%s]", added, idstr);
        }
    }

    LOGI_L(CAT_TOKENIZER, 0, "BPE 完成: 共 %d 个 token", n_out);
    free(starts); free(ends);
    return n_out;
}

/* ============ 解码 ============ */

int tokenizer_decode(Tokenizer *t, int id, char *buf, int buf_cap) {
    /* id -> vocab 字符串 -> 逐字符还原成原始字节。
     *
     * byte-level 字符 -> 原始字节的反向映射:
     *   vocab 字符串里每个 unicode "字符"(可能 1-2 字节 UTF-8)
     *   对应一个原始字节。我们要建立 byte_map 的反向表。
     *
     * 简化做法: vocab 字符串 = 若干个 byte-level 字符拼接。
     *           我们用 byte_map 的 256 条建立 "字符串 -> 原始字节" 反查,
     *           然后从 vocab 字符串头部依次匹配每个字符, 还原字节。
     */
    if (id < 0 || id >= t->vocab_size) {
        /* 可能是 special token */
        for (int i = 0; i < t->n_special; i++) {
            if (t->special_ids[i] == id) {
                int l = t->special_lens[i];
                if (l > buf_cap - 1) l = buf_cap - 1;
                memcpy(buf, t->special_strs[i], l);
                return l;
            }
        }
        return 0;
    }

    /* 检查是否是 special token (id 在 special 范围) */
    for (int i = 0; i < t->n_special; i++) {
        if (t->special_ids[i] == id) {
            int l = t->special_lens[i];
            if (l > buf_cap - 1) l = buf_cap - 1;
            memcpy(buf, t->special_strs[i], l);
            return l;
        }
    }

    const char *s = t->vocab[id];
    int slen = t->vocab_len[id];
    int oi = 0;
    int si = 0;
    while (si < slen && oi < buf_cap - 1) {
        /* 当前 byte-level 字符的 UTF-8 长度: 看首字节 */
        unsigned char c0 = (unsigned char)s[si];
        int clen = (c0 < 0x80) ? 1 : (c0 < 0xE0) ? 2 : (c0 < 0xF0) ? 3 : 4;
        if (si + clen > slen) break;
        /* 在 byte_map 里找匹配的原始字节 */
        int found = -1;
        for (int b = 0; b < 256; b++) {
            if (t->byte_map_len[b] == clen &&
                memcmp(t->byte_map_chars[b], s + si, clen) == 0) {
                found = b;
                break;
            }
        }
        if (found >= 0) {
            buf[oi++] = (char)found;
        } else {
            /* 找不到就原样输出(极少见) */
            buf[oi++] = s[si];
        }
        si += clen;
    }
    return oi;
}
