/* json.c — 极简 JSON 解析器
 *
 * 实现思路: 递归下降。指针 p 在文本里前进，遇到 '{' '[' 开始递归。
 * 内存: 所有节点用 malloc，构成树；json_free 递归释放。
 *
 * 这里只支持 safetensors header 用到的子集:
 *   object / array / string / number(int 或 float) / true/false/null
 * 不支持注释、不支持尾随逗号的容错(但 safetensors 不需要)。
 */

#include "json.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* 解析上下文: 当前指针 + 错误标志 */
typedef struct {
    const char *p;     /* 当前读取位置 */
    int error;
} Parser;

static JsonValue *parse_value(Parser *ps);

/* 跳过空白 */
static void skip_ws(Parser *ps) {
    while (*ps->p && isspace((unsigned char)*ps->p)) ps->p++;
}

static JsonValue *new_value(JsonType t) {
    JsonValue *v = (JsonValue *)calloc(1, sizeof(JsonValue));
    if (v) v->type = t;
    return v;
}

/* 解析字符串(已消费了开头的 '"')。
 * 处理常见转义: \" \\ \/ \b \f \n \r \t \uXXXX
 * \u 的代理对这里简化处理(直接存 BMP 码点，safetensors 里罕见)。
 */
static char *parse_string_raw(Parser *ps) {
    /* 第一遍: 算长度 */
    const char *start = ps->p;
    int len = 0;
    while (*ps->p && *ps->p != '"') {
        if (*ps->p == '\\') {
            ps->p++;
            if (*ps->p == 'u') {
                ps->p += 4;       /* \uXXXX 后面 4 个十六进制 */
                len += 3;         /* UTF-8 最多 3 字节(BMP) */
            } else {
                len++;
                ps->p++;
            }
        } else {
            len++;
            ps->p++;
        }
    }
    if (*ps->p != '"') { ps->error = 1; return NULL; }

    /* 第二遍: 填内容 */
    char *out = (char *)malloc(len + 1);
    int oi = 0;
    ps->p = start;
    while (*ps->p && *ps->p != '"') {
        if (*ps->p == '\\') {
            ps->p++;
            char c = *ps->p;
            char repl = 0;
            switch (c) {
                case '"':  repl = '"';  break;
                case '\\': repl = '\\'; break;
                case '/':  repl = '/';  break;
                case 'b':  repl = '\b'; break;
                case 'f':  repl = '\f'; break;
                case 'n':  repl = '\n'; break;
                case 'r':  repl = '\r'; break;
                case 't':  repl = '\t'; break;
                case 'u': {
                    /* 解析 4 位十六进制 -> 码点 -> UTF-8 */
                    unsigned int cp = 0;
                    for (int i = 0; i < 4; i++) {
                        ps->p++;
                        char hc = *ps->p;
                        cp <<= 4;
                        if (hc >= '0' && hc <= '9') cp |= hc - '0';
                        else if (hc >= 'a' && hc <= 'f') cp |= hc - 'a' + 10;
                        else if (hc >= 'A' && hc <= 'F') cp |= hc - 'A' + 10;
                    }
                    if (cp < 0x80) {
                        out[oi++] = (char)cp;
                    } else if (cp < 0x800) {
                        out[oi++] = (char)(0xC0 | (cp >> 6));
                        out[oi++] = (char)(0x80 | (cp & 0x3F));
                    } else {
                        out[oi++] = (char)(0xE0 | (cp >> 12));
                        out[oi++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                        out[oi++] = (char)(0x80 | (cp & 0x3F));
                    }
                    ps->p++;  /* 跳过最后一个十六进制字符 */
                    continue;
                }
                default:
                    repl = c;
            }
            out[oi++] = repl;
            ps->p++;
        } else {
            out[oi++] = *ps->p++;
        }
    }
    /* 跳过闭合引号 */
    if (*ps->p == '"') ps->p++;
    out[oi] = 0;
    return out;
}

static JsonValue *parse_string(Parser *ps) {
    if (*ps->p != '"') { ps->error = 1; return NULL; }
    ps->p++;  /* 跳过开引号 */
    char *s = parse_string_raw(ps);
    if (!s) return NULL;
    JsonValue *v = new_value(JSON_STRING);
    v->string = s;
    return v;
}

static JsonValue *parse_number(Parser *ps) {
    char *end;
    double d = strtod(ps->p, &end);
    if (end == ps->p) { ps->error = 1; return NULL; }
    ps->p = end;
    JsonValue *v = new_value(JSON_NUMBER);
    v->number = d;
    return v;
}

/* 解析 true/false/null */
static JsonValue *parse_literal(Parser *ps) {
    if (strncmp(ps->p, "true", 4) == 0) {
        ps->p += 4;
        JsonValue *v = new_value(JSON_BOOL);
        v->boolean = 1;
        return v;
    }
    if (strncmp(ps->p, "false", 5) == 0) {
        ps->p += 5;
        JsonValue *v = new_value(JSON_BOOL);
        v->boolean = 0;
        return v;
    }
    if (strncmp(ps->p, "null", 4) == 0) {
        ps->p += 4;
        return new_value(JSON_NULL);
    }
    ps->error = 1;
    return NULL;
}

static JsonValue *parse_array(Parser *ps) {
    ps->p++;  /* 跳过 '[' */
    JsonValue *arr = new_value(JSON_ARRAY);
    JsonValue *tail = NULL;
    skip_ws(ps);
    if (*ps->p == ']') { ps->p++; return arr; }   /* 空数组 */
    for (;;) {
        JsonValue *elem = parse_value(ps);
        if (!elem) { json_free(arr); return NULL; }
        if (tail) tail->next_sibling = elem;
        else arr->first_child = elem;
        tail = elem;
        skip_ws(ps);
        if (*ps->p == ',') { ps->p++; skip_ws(ps); continue; }
        if (*ps->p == ']') { ps->p++; break; }
        ps->error = 1; json_free(arr); return NULL;
    }
    return arr;
}

static JsonValue *parse_object(Parser *ps) {
    ps->p++;  /* 跳过 '{' */
    JsonValue *obj = new_value(JSON_OBJECT);
    JsonValue *tail = NULL;
    skip_ws(ps);
    if (*ps->p == '}') { ps->p++; return obj; }   /* 空对象 */
    for (;;) {
        skip_ws(ps);
        if (*ps->p != '"') { ps->error = 1; json_free(obj); return NULL; }
        ps->p++;
        char *key = parse_string_raw(ps);
        if (!key) { json_free(obj); return NULL; }
        /* key 节点 */
        JsonValue *kn = new_value(JSON_STRING);
        kn->string = key;
        skip_ws(ps);
        if (*ps->p != ':') { ps->error = 1; json_free(kn); json_free(obj); return NULL; }
        ps->p++;
        skip_ws(ps);
        JsonValue *val = parse_value(ps);
        if (!val) { json_free(kn); json_free(obj); return NULL; }
        kn->value_child = val;
        if (tail) tail->next_sibling = kn;
        else obj->first_child = kn;
        tail = kn;
        skip_ws(ps);
        if (*ps->p == ',') { ps->p++; continue; }
        if (*ps->p == '}') { ps->p++; break; }
        ps->error = 1; json_free(obj); return NULL;
    }
    return obj;
}

static JsonValue *parse_value(Parser *ps) {
    skip_ws(ps);
    char c = *ps->p;
    if (c == '{') return parse_object(ps);
    if (c == '[') return parse_array(ps);
    if (c == '"') return parse_string(ps);
    if (c == '-' || (c >= '0' && c <= '9')) return parse_number(ps);
    if (c == 't' || c == 'f' || c == 'n') return parse_literal(ps);
    ps->error = 1;
    return NULL;
}

JsonValue *json_parse(const char *text) {
    Parser ps = { text, 0 };
    JsonValue *v = parse_value(&ps);
    if (ps.error) {
        if (v) json_free(v);
        return NULL;
    }
    return v;
}

void json_free(JsonValue *v) {
    if (!v) return;
    /* 先释放子树 */
    JsonValue *c = v->first_child;
    while (c) {
        JsonValue *nxt = c->next_sibling;
        /* 如果是 object 的 key 节点，还要释放它的 value_child */
        if (v->type == JSON_OBJECT && c->value_child) {
            json_free(c->value_child);
        }
        json_free(c);
        c = nxt;
    }
    if (v->string) free(v->string);
    free(v);
}

JsonValue *json_object_get(JsonValue *obj, const char *key) {
    if (!obj || obj->type != JSON_OBJECT) return NULL;
    for (JsonValue *c = obj->first_child; c; c = c->next_sibling) {
        if (c->type == JSON_STRING && c->string && strcmp(c->string, key) == 0) {
            return c->value_child;
        }
    }
    return NULL;
}

int json_array_len(JsonValue *arr) {
    if (!arr || arr->type != JSON_ARRAY) return 0;
    int n = 0;
    for (JsonValue *c = arr->first_child; c; c = c->next_sibling) n++;
    return n;
}

JsonValue *json_array_get(JsonValue *arr, int i) {
    if (!arr || arr->type != JSON_ARRAY) return NULL;
    int n = 0;
    for (JsonValue *c = arr->first_child; c; c = c->next_sibling, n++) {
        if (n == i) return c;
    }
    return NULL;
}
