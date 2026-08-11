# 第三章 JSON 解析器

> 第 3 章 | 配合源码：`json.h` / `json.c`（约 330 行） | [上一章：权重的存储与加载](./02-weights.md) | [下一章：模型结构](./04-model.md)

本章解决一个问题：**safetensors 的 header 是 JSON，怎么在纯 C 里解析它**。

上一章我们看到，safetensors 文件的第二段是一个 JSON 对象，记录了每个张量的名字、dtype、shape、data_offsets。引擎要拿到这些信息才能去 raw buffer 里取数据。问题是——本项目不依赖任何外部库，而 C 标准库没有 JSON 解析器。所以我们必须自己写一个。

听起来很吓人，但其实 JSON 的语法极其简单，一个 300 行的递归下降解析器就够用了。

---

## 为什么要自己写 JSON 解析器

safetensors 的 header 是一个 JSON 对象，长这样（简化）：

```json
{
  "model.embed_tokens.weight": {
    "dtype": "BF16",
    "shape": [151936, 896],
    "data_offsets": [0, 272269312]
  },
  "model.layers.0.self_attn.q_proj.weight": {
    "dtype": "BF16",
    "shape": [896, 896],
    "data_offsets": [272269312, 273674368]
  },
  ...
  "__metadata__": { "format": "pt" }
}
```

解析它需要 JSON parser。但我们不依赖任何外部库（这是本项目"纯 C、零依赖"的设计原则），所以手写一个**刚好够用**的极简版。

**够用**的标准是：能解析 safetensors header 里出现的所有结构——对象、数组、字符串、数字、`true`/`false`/`null`。不追求完整 RFC 兼容，不支持注释、不支持尾随逗号的容错（但 safetensors 不需要这些）。

> 📍 在哪：`json.h` 顶部注释明确写了"目标：刚好够解析 safetensors 的 header。不追求完整 RFC 兼容"。

## 递归下降解析（Recursive Descent）

这是整个解析器的核心思想。

- JSON 是嵌套结构（对象里有数组，数组里有对象……）。递归下降 = 遇到 `{` 就调 `parse_object()`，遇到 `[` 就调 `parse_array()`，函数自己调自己。
- 为什么需要：嵌套结构没法用简单的循环处理，递归天然契合——每深入一层就多一次函数调用，栈自动记录"我在哪一层"。

整个解析器入口是 `parse_value`，它根据当前字符决定走哪个分支：

```
parse_value:
  if '{' → parse_object   (里面又调 parse_value 解析每个值)
  if '[' → parse_array    (里面又调 parse_value 解析每个元素)
  if '"' → parse_string
  if 数字 → parse_number
  if t/f/n → parse_literal (true/false/null)
```

对应的 C 代码非常直白（`json.c`）：

```c
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
```

`Parser` 结构体只有一个指针和一个错误标志，游标 `p` 在文本里逐字符前进：

```c
typedef struct {
    const char *p;     /* 当前读取位置 */
    int error;
} Parser;
```

### 几个子解析器怎么工作

**parse_object**（解析 `{...}`）：跳过 `{`，然后循环「读 key 字符串 → 跳过 `:` → 读 value → 跳过 `,` 或遇到 `}`」。

```c
static JsonValue *parse_object(Parser *ps) {
    ps->p++;                        // 跳过 '{'
    JsonValue *obj = new_value(JSON_OBJECT);
    ...
    for (;;) {
        // 读 key (必须是字符串)
        char *key = parse_string_raw(ps);
        // 跳过 ':'
        ...
        // 读 value (递归调 parse_value)
        JsonValue *val = parse_value(ps);   // ← 递归点
        kn->value_child = val;              // key 指向 value
        ...
        if (*ps->p == ',') { ps->p++; continue; }
        if (*ps->p == '}') { ps->p++; break; }
    }
    return obj;
}
```

**parse_array**（解析 `[...]`）：同理，跳过 `[`，循环读每个元素（每个元素都递归调 `parse_value`），遇到 `]` 结束。

**parse_number**（解析数字）：直接用 C 标准库的 `strtod`，它能处理整数、小数、科学计数法。`strtod` 会通过 `end` 指针告诉你它读到了哪里。

```c
static JsonValue *parse_number(Parser *ps) {
    char *end;
    double d = strtod(ps->p, &end);
    if (end == ps->p) { ps->error = 1; return NULL; }
    ps->p = end;
    ...
}
```

**parse_literal**（解析 `true`/`false`/`null`）：用 `strncmp` 比较前几个字符，匹配上就跳过对应长度。

### 相关术语

| 术语 | 含义 |
|------|------|
| **RFC** | 互联网标准文档（如 RFC 8259 = JSON 标准）。我们的 parser 不追求完整 RFC 兼容 |
| **反转义 (unescape)** | 把 JSON 字符串里的 `\"` 还原成 `"`、`\n` 还原成换行等 |
| **代理对 (surrogate pair)** | UTF-16 用两个 16 位编码表示一个罕见字符（如 emoji）。我们简化处理 |
| **BMP (Basic Multilingual Plane)** | Unicode 的基本平面，码点 0-65535 |

### 字符串解析的两遍法

字符串是最复杂的部分，因为有转义。`parse_string_raw` 用了"两遍法"：

1. 第一遍：扫描一遍算出反转义后的长度（因为 `\"` 两个字符变成一个 `"`，长度会变）。
2. **malloc** 分配刚好够的内存。
3. 第二遍：再扫一遍，真正填内容，边扫边反转义。

```c
static char *parse_string_raw(Parser *ps) {
    /* 第一遍: 算长度 */
    const char *start = ps->p;
    int len = 0;
    while (*ps->p && *ps->p != '"') {
        if (*ps->p == '\\') { ... 处理转义, 累加 len ... }
        else { len++; ps->p++; }
    }
    /* 第二遍: 填内容 */
    char *out = (char *)malloc(len + 1);
    ...
}
```

支持的转义：`\" \\ \/ \b \f \n \r \t`，以及 `\uXXXX`（把 4 位十六进制码点转成 UTF-8 字节）。`\u` 的代理对做了简化处理——直接存 BMP 码点，因为 safetensors header 里基本不会出现 emoji 之类需要代理对的字符。

> 📍 在哪：`json.c` 的 `parse_string_raw()`，两遍扫描的逻辑都在里面。

## JsonValue 树结构

解析完之后，整段 JSON 文本变成内存里的一棵树，每个节点是一个 `JsonValue`：

```c
typedef struct JsonValue {
    JsonType type;           // NULL/BOOL/NUMBER/STRING/ARRAY/OBJECT
    double number;           // type==NUMBER 时的值
    int boolean;             // type==BOOL 时的值
    char *string;            // type==STRING 时的值

    /* 树结构: 用兄弟链表表示 */
    struct JsonValue *first_child;   // 第一个子节点
    struct JsonValue *next_sibling;  // 下一个兄弟
    struct JsonValue *value_child;   // OBJECT 的 key 指向 value
} JsonValue;
```

`type` 是个枚举，定义在 `json.h`：

```c
typedef enum {
    JSON_NULL,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,    /* 字符串值，已反转义 */
    JSON_ARRAY,     /* 元素在 first_child 链表里 */
    JSON_OBJECT     /* 键值对: 用 first_child 遍历 (key 是 JsonValue(JSON_STRING)) */
} JsonType;
```

设计要点：数组和对象都用「FirstChild + NextSibling」的链表表示，避免动态数组。

```
JSON:  { "a": 1, "b": [2, 3] }

树:
  OBJECT
    │ first_child
    ▼
  STRING("a") ──next_sibling──► STRING("b")
    │ value_child                   │ value_child
    ▼                               ▼
  NUMBER(1)                       ARRAY
                                   │ first_child
                                   ▼
                                 NUMBER(2) ──next_sibling──► NUMBER(3)
```

为什么用链表不用动态数组？因为 safetensors header 里的对象通常只有几个字段（`dtype`/`shape`/`data_offsets`），数组最多几维（shape 一般 1-2 维），链表的内存分配更简单，不用写 `realloc` 扩容逻辑。

### OBJECT 的"key 节点"技巧

注意 OBJECT 的存储方式有点特别：key 本身也是一个 `JsonValue`（类型 `JSON_STRING`），它的 `value_child` 指向真正的 value。这样 OBJECT 的 `first_child` 链表里混着的全是 key 节点，遍历时每个 key 通过 `value_child` 拿到 value。

这是为了复用同一套"链表遍历"代码——object 和 array 的子节点管理完全一样，区别只在 object 的子节点（key）多挂了一个 `value_child`。

### 三个查询函数

解析成树之后，对外提供三个查询接口：

```c
/* 在 OBJECT 中按 key 查子节点，返回 value(不是 key 节点)。找不到返回 NULL。 */
JsonValue *json_object_get(JsonValue *obj, const char *key);

/* 取 ARRAY 长度 */
int json_array_len(JsonValue *arr);

/* 取 ARRAY 第 i 个元素 */
JsonValue *json_array_get(JsonValue *arr, int i);
```

`json_object_get` 就是线性扫描 `first_child` 链表，逐个比较 key 字符串：

```c
JsonValue *json_object_get(JsonValue *obj, const char *key) {
    if (!obj || obj->type != JSON_OBJECT) return NULL;
    for (JsonValue *c = obj->first_child; c; c = c->next_sibling) {
        if (c->type == JSON_STRING && c->string && strcmp(c->string, key) == 0) {
            return c->value_child;   // 注意返回的是 value, 不是 key
        }
    }
    return NULL;
}
```

对 safetensors header 这种几十个字段的小对象，线性查找完全够快，不需要哈希表。

## 在 safetensors 里怎么用

第二章的 `safetensors_open()` 会在打开文件时调用一次 `json_parse`，把 header 文本解析成树，缓存到 `st->header`：

```c
st->header = json_parse(hdr);
```

之后 `safetensors_find()` 查任何张量，都走这棵已经建好的树，不用重复解析：

```c
JsonValue *t = json_object_get(root, name);          // 找到这个张量的节点
JsonValue *dtype = json_object_get(t, "dtype");      // 在它里面找 dtype
JsonValue *shape = json_object_get(t, "shape");      // 找 shape (是个 array)
JsonValue *doff  = json_object_get(t, "data_offsets");

unsigned long off0 = (unsigned long)json_array_get(doff, 0)->number;  // [start, end)
unsigned long off1 = (unsigned long)json_array_get(doff, 1)->number;
```

这就是 `json.c` 存在的全部意义——给 `safetensors.c` 提供"按名查张量元信息"的能力。

## 内存管理

所有节点都用 `malloc`/`calloc` 分配，构成一棵树。`json_free` 递归释放整棵树：

```c
void json_free(JsonValue *v) {
    if (!v) return;
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
```

注意 OBJECT 节点释放时的细节：它的 `first_child` 链表里是 key 节点，每个 key 节点的 `value_child` 指向 value——这个 value 要单独 `json_free`，否则内存泄漏。ARRAY 节点则没有这个 `value_child`，直接递归子节点就行。

> 📍 在哪：`json.c` 的 `json_free()`，被 `safetensors_close()` 调用，在关闭文件时把 header 树整个释放掉。

## 错误处理

解析器的错误处理很朴素：`Parser` 里有个 `error` 标志，任何子解析器发现不对劲（比如对象里 key 后面不是 `:`，或者数字 `strtod` 解析失败）就把 `error` 置 1，返回 `NULL`。上层 `parse_value` 拿到 `NULL` 就一路返回 `NULL`，最终 `json_parse` 发现 `error` 就丢弃半成品、返回 `NULL`。

```c
JsonValue *json_parse(const char *text) {
    Parser ps = { text, 0 };
    JsonValue *v = parse_value(&ps);
    if (ps.error) {
        if (v) json_free(v);   // 解析到一半也要清理
        return NULL;
    }
    return v;
}
```

`json_free` 在中途失败时也能正确清理已分配的子树——这就是为什么每个 `parse_xxx` 在出错时都会先 `json_free` 已经建好的部分再返回 `NULL`，避免内存泄漏。

这种"出错就回滚"的策略对 safetensors header 足够用——header 格式是程序生成的，正常情况不会出错；如果真出错，直接报错退出，不需要"容错恢复"。

---

## 本章小结

1. 为什么自己写 JSON 解析器：本项目零依赖，C 标准库没有 JSON，而 safetensors header 是 JSON。
2. 递归下降：遇到 `{` 调 `parse_object`，遇到 `[` 调 `parse_array`，函数自己调自己，天然契合 JSON 的嵌套结构。
3. JsonValue 树：解析结果是一棵树，用「FirstChild + NextSibling」链表表示数组和对象的子节点；OBJECT 的 key 节点通过 `value_child` 指向 value。
4. 字符串两遍法：第一遍算反转义后的长度，第二遍填内容，避免缓冲区大小估计错误。
5. 生命周期：`safetensors_open` 时 `json_parse` 一次建树，之后反复查询；`safetensors_close` 时 `json_free` 整棵释放。

这个解析器虽然只有 300 行，但它是一个完整的、能正确处理真实 safetensors header 的 JSON 解析器。理解了它，你也就理解了所有递归下降解析器（包括很多编译器前端）的基本套路。

> 返回：[第二章 权重的存储与加载](02-weights.md) | [第一章 基础概念](01-basics.md)
