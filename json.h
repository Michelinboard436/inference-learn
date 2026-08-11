#ifndef JSON_H
#define JSON_H

/*
 * json.h — 极简 JSON 解析器
 *
 * 目标: 刚好够解析 safetensors 的 header。
 * 不追求完整 RFC 兼容，不支持浮点科学计数法转义等边角情况。
 *
 * 解析策略: 把整个 JSON 字符串解析成一棵树 (JsonValue 数组)，
 * 之后可以反复查询。safetensors header 通常只有几十个张量，内存可忽略。
 */

typedef enum {
    JSON_NULL,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,    /* 字符串值，已反转义 */
    JSON_ARRAY,     /* 元素在 .items[] */
    JSON_OBJECT     /* 键值对: 用 first_child 遍历 (key 是 JsonValue(JSON_STRING)) */
} JsonType;

typedef struct JsonValue {
    JsonType type;
    /* 标量值 */
    double number;
    int boolean;
    char *string;       /* type==STRING 或 OBJECT 的 key 用的字符串 */

    /* 数组/对象的子节点链表 */
    struct JsonValue *first_child;
    struct JsonValue *next_sibling;  /* 同一个父下的下一个 */

    /* 仅 OBJECT: first_child 指向的节点是 key(STRING)，
       它的 value_child 指向对应的 value */
    struct JsonValue *value_child;
} JsonValue;

/* 解析 JSON 文本，返回根节点(通常是 OBJECT)。失败返回 NULL。
 * 返回的树用 json_free 释放。 */
JsonValue *json_parse(const char *text);

/* 释放整棵树 */
void json_free(JsonValue *v);

/* 在 OBJECT 中按 key 查子节点，返回 value(不是 key 节点)。找不到返回 NULL。 */
JsonValue *json_object_get(JsonValue *obj, const char *key);

/* 取 ARRAY 长度 */
int json_array_len(JsonValue *arr);

/* 取 ARRAY 第 i 个元素 */
JsonValue *json_array_get(JsonValue *arr, int i);

#endif // JSON_H
