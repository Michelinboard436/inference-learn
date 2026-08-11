> 🌐 [中文版](../cn/03-json) | English

# Chapter 3 · JSON Parser

> Chapter 3 | Companion source: `json.h` / `json.c` (~330 lines) | [Previous: Weight Storage & Loading](./02-weights.md) | [Next: Model Architecture](./04-model.md)

This chapter answers one question: **the safetensors header is JSON — how do we parse it in pure C**.

In the previous chapter we saw that the second segment of a safetensors file is a JSON object recording each tensor's name, dtype, shape, and data_offsets. The engine has to get hold of this information before it can fetch data from the raw buffer. The problem is — this project depends on no external libraries, and the C standard library has no JSON parser. So we have to write one ourselves.

It sounds intimidating, but JSON's syntax is so simple that a 300-line recursive descent parser is more than enough.

---

## Why Write Your Own JSON Parser

The safetensors header is a JSON object that looks like this (simplified):

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

Parsing it needs a JSON parser. But we don't depend on any external library (this is this project's "pure C, zero-dependency" design principle), so we hand-write a minimalist version that is **just enough**.

"Just enough" means: able to parse every structure that appears in a safetensors header — objects, arrays, strings, numbers, `true`/`false`/`null`. It does not aim for full RFC compliance; it doesn't support comments or tolerate trailing commas (but safetensors doesn't need any of those).

> 📍 **Where**: the top comment of `json.h` explicitly states the goal: "just enough to parse the safetensors header. Does not aim for full RFC compliance."

## Recursive Descent Parsing

This is the core idea of the entire parser.

- Plain English**: JSON is a nested structure (objects inside arrays inside objects...). Recursive descent = when you hit `{` you call `parse_object()`; when you hit `[` you call `parse_array()`; the functions call themselves.
- Why it's needed**: a nested structure can't be handled by a simple loop; recursion fits naturally — each level deeper means one more function call, and the stack automatically records "which level am I on."

The whole parser's entry point is `parse_value`, which branches based on the current character:

```
parse_value:
  if '{' → parse_object   (which in turn calls parse_value to parse each value)
  if '[' → parse_array    (which in turn calls parse_value to parse each element)
  if '"' → parse_string
  if digit → parse_number
  if t/f/n → parse_literal (true/false/null)
```

The corresponding C code is very straightforward (`json.c`):

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

The `Parser` struct holds just one pointer and an error flag; the cursor `p` advances character by character through the text:

```c
typedef struct {
    const char *p;     /* current read position */
    int error;
} Parser;
```

### How the Sub-Parsers Work

**parse_object** (parses `{...}`): skips `{`, then loops "read a key string → skip `:` → read a value → skip `,` or hit `}`".

```c
static JsonValue *parse_object(Parser *ps) {
    ps->p++;                        // skip '{'
    JsonValue *obj = new_value(JSON_OBJECT);
    ...
    for (;;) {
        // read the key (must be a string)
        char *key = parse_string_raw(ps);
        // skip ':'
        ...
        // read the value (recursive call to parse_value)
        JsonValue *val = parse_value(ps);   // ← recursion point
        kn->value_child = val;              // key points to value
        ...
        if (*ps->p == ',') { ps->p++; continue; }
        if (*ps->p == '}') { ps->p++; break; }
    }
    return obj;
}
```

**parse_array** (parses `[...]`): same idea — skip `[`, loop reading each element (each element recursively calls `parse_value`), end on `]`.

**parse_number** (parses a number): directly uses the C standard library's `strtod`, which handles integers, decimals, and scientific notation. `strtod` tells you how far it read via the `end` pointer.

```c
static JsonValue *parse_number(Parser *ps) {
    char *end;
    double d = strtod(ps->p, &end);
    if (end == ps->p) { ps->error = 1; return NULL; }
    ps->p = end;
    ...
}
```

**parse_literal** (parses `true`/`false`/`null`): uses `strncmp` to compare the first few characters, and if it matches, advances by the corresponding length.

### Related Terminology

| Term | Meaning |
|------|------|
| **RFC** | Internet standard document (e.g., RFC 8259 = the JSON standard). Our parser doesn't aim for full RFC compliance |
| **unescape** | Restore `\"` inside a JSON string back to `"`, `\n` back to a newline, etc. |
| **surrogate pair** | UTF-16 uses two 16-bit code units to represent a rare character (like an emoji). We handle this in a simplified way |
| **BMP (Basic Multilingual Plane)** | Unicode's basic plane, code points 0-65535 |

### The Two-Pass Method for String Parsing

Strings are the trickiest part because of escapes. `parse_string_raw` uses a "two-pass" method:

1. **First pass**: scan once to compute the unescaped length (because `\"`, two characters, becomes one `"`, which changes the length).
2. **malloc** exactly enough memory.
3. **Second pass**: scan again, actually filling in the content while unescaping on the fly.

```c
static char *parse_string_raw(Parser *ps) {
    /* first pass: compute length */
    const char *start = ps->p;
    int len = 0;
    while (*ps->p && *ps->p != '"') {
        if (*ps->p == '\\') { ... handle escape, accumulate len ... }
        else { len++; ps->p++; }
    }
    /* second pass: fill content */
    char *out = (char *)malloc(len + 1);
    ...
}
```

Supported escapes: `\" \\ \/ \b \f \n \r \t`, plus `\uXXXX` (converts a 4-digit hex code point into UTF-8 bytes). The `\u` surrogate pair is handled in a simplified way — it stores the BMP code point directly, because safetensors headers essentially never contain characters like emoji that would need a surrogate pair.

> 📍 **Where**: `parse_string_raw()` in `json.c`; both passes of the scan live inside it.

## The JsonValue Tree Structure

Once parsing is done, the entire JSON text has become a tree in memory, where each node is a `JsonValue`:

```c
typedef struct JsonValue {
    JsonType type;           // NULL/BOOL/NUMBER/STRING/ARRAY/OBJECT
    double number;           // the value when type==NUMBER
    int boolean;             // the value when type==BOOL
    char *string;            // the value when type==STRING

    /* tree structure: represented as a sibling linked list */
    struct JsonValue *first_child;   // first child
    struct JsonValue *next_sibling;  // next sibling
    struct JsonValue *value_child;   // an OBJECT's key points to its value
} JsonValue;
```

`type` is an enum, defined in `json.h`:

```c
typedef enum {
    JSON_NULL,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,    /* string value, already unescaped */
    JSON_ARRAY,     /* elements are in the first_child list */
    JSON_OBJECT     /* key-value pairs: traverse via first_child (the key is a JsonValue(JSON_STRING)) */
} JsonType;
```

**Design point**: both arrays and objects are represented with a "FirstChild + NextSibling" linked list, avoiding dynamic arrays.

```
JSON:  { "a": 1, "b": [2, 3] }

Tree:
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

Why a linked list instead of a dynamic array? Because the objects in a safetensors header usually have only a few fields (`dtype`/`shape`/`data_offsets`), and arrays are at most a few elements (shape is usually 1-2 dimensions); a linked list makes memory allocation simpler and avoids writing `realloc` growth logic.

### The OBJECT "Key Node" Trick

Note that the way an OBJECT is stored is a little special: the key itself is also a `JsonValue` (of type `JSON_STRING`), and its `value_child` points to the actual value. This way the OBJECT's `first_child` list is uniformly a list of key nodes, and when traversing, each key reaches its value through `value_child`.

This is to reuse the same "list traversal" code — the child management of an object and an array is identical; the only difference is that an object's children (keys) carry an extra `value_child`.

### Three Query Functions

Once parsed into a tree, three query interfaces are exposed:

```c
/* Look up a child by key inside an OBJECT; returns the value (not the key node). Returns NULL if not found. */
JsonValue *json_object_get(JsonValue *obj, const char *key);

/* Get the length of an ARRAY */
int json_array_len(JsonValue *arr);

/* Get the i-th element of an ARRAY */
JsonValue *json_array_get(JsonValue *arr, int i);
```

`json_object_get` just does a linear scan of the `first_child` list, comparing key strings one by one:

```c
JsonValue *json_object_get(JsonValue *obj, const char *key) {
    if (!obj || obj->type != JSON_OBJECT) return NULL;
    for (JsonValue *c = obj->first_child; c; c = c->next_sibling) {
        if (c->type == JSON_STRING && c->string && strcmp(c->string, key) == 0) {
            return c->value_child;   // note: returns the value, not the key
        }
    }
    return NULL;
}
```

For a small object with a few dozen fields like a safetensors header, linear lookup is plenty fast — no hash table needed.

## How It's Used Inside Safetensors

In Chapter 2, `safetensors_open()` calls `json_parse` once when opening the file, parses the header text into a tree, and caches it on `st->header`:

```c
st->header = json_parse(hdr);
```

After that, every `safetensors_find()` lookup for any tensor goes through this already-built tree, with no re-parsing:

```c
JsonValue *t = json_object_get(root, name);          // find this tensor's node
JsonValue *dtype = json_object_get(t, "dtype");      // find dtype inside it
JsonValue *shape = json_object_get(t, "shape");      // find shape (an array)
JsonValue *doff  = json_object_get(t, "data_offsets");

unsigned long off0 = (unsigned long)json_array_get(doff, 0)->number;  // [start, end)
unsigned long off1 = (unsigned long)json_array_get(doff, 1)->number;
```

That's the entire reason `json.c` exists — to give `safetensors.c` the ability to "look up tensor metadata by name."

## Memory Management

All nodes are allocated with `malloc`/`calloc` and form a tree. `json_free` recursively frees the whole tree:

```c
void json_free(JsonValue *v) {
    if (!v) return;
    JsonValue *c = v->first_child;
    while (c) {
        JsonValue *nxt = c->next_sibling;
        /* if this is an object's key node, also free its value_child */
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

Note the detail when freeing an OBJECT node: its `first_child` list holds key nodes, and each key node's `value_child` points to a value — that value must be `json_free`-d separately, or you leak memory. An ARRAY node has no such `value_child`, so you just recurse into the children directly.

> 📍 **Where**: `json_free()` in `json.c`, called by `safetensors_close()` to free the entire header tree when the file is closed.

## Error Handling

The parser's error handling is plain: there's an `error` flag in `Parser`, and any sub-parser that notices something wrong (for example the token after a key in an object isn't `:`, or `strtod` fails on a number) sets `error` to 1 and returns `NULL`. The caller `parse_value` gets `NULL` and returns `NULL` all the way up; eventually `json_parse` sees `error`, discards the half-built result, and returns `NULL`.

```c
JsonValue *json_parse(const char *text) {
    Parser ps = { text, 0 };
    JsonValue *v = parse_value(&ps);
    if (ps.error) {
        if (v) json_free(v);   // clean up even if we parsed only halfway
        return NULL;
    }
    return v;
}
```

`json_free` can also correctly clean up already-allocated sub-trees on a mid-way failure — that's why every `parse_xxx` function, on error, first `json_free`s the part it has already built before returning `NULL`, to avoid memory leaks.

This "rollback on error" strategy is sufficient for a safetensors header — the header format is program-generated, so under normal conditions it won't be malformed; if it truly is malformed, we just error out and exit, with no need for "fault-tolerant recovery."

---

## Chapter Summary

1. **Why write your own JSON parser**: this project is zero-dependency, the C standard library has no JSON, and the safetensors header is JSON.
2. **Recursive descent**: on `{` call `parse_object`, on `[` call `parse_array`, the functions call themselves — a natural fit for JSON's nested structure.
3. **The JsonValue tree**: the parse result is a tree, with array and object children represented by a "FirstChild + NextSibling" list; an OBJECT's key node points to its value via `value_child`.
4. **The two-pass string method**: the first pass computes the unescaped length, the second pass fills the content, avoiding buffer-size miscalculation.
5. **Lifecycle**: `safetensors_open` calls `json_parse` once to build the tree, which is then queried repeatedly; `safetensors_close` calls `json_free` to free the whole tree.

Although this parser is only 300 lines, it is a complete JSON parser that correctly handles real safetensors headers. Once you understand it, you also understand the basic pattern of every recursive descent parser (including many compiler front-ends).

> Back to: [Chapter 2 · Weight Storage & Loading](02-weights.md) | [Chapter 1 · Basics](01-basics.md)
