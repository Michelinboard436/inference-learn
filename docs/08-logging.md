# 第 8 章 日志与可视化（trace.c / report.c）

> 第 8 章 | [上一章：第 7 章 采样与生成](./07-sampling.md) | [下一章：第 9 章 调试与验证方法论](./09-debugging.md)

> 对应源码：`trace.h` / `trace.c`（~210 行）+ `report.h` / `report.c`（~145 行）
> 这章回答一个问题：**引擎跑起来后，怎么看见内部每一步发生了什么。**

推理引擎最让人头疼的不是「跑不起来」，而是「跑起来但输出是垃圾，却不知道哪一步错了」。
本章讲的日志和可视化系统，就是让你能**看见**矩阵乘法、归一化、注意力每一步的数值。

---

## 目录

- [分级日志：info / debug / trace](#分级日志info--debug--trace)
- [stderr vs stdout：日志和生成分开](#stderr-vs-stdout日志和生成分开)
- [TTY 检测与 ANSI 颜色](#tty-检测与-ansi-颜色)
- [向量统计：范数、激活率](#向量统计范数激活率)
- [HTML 报告](#html-报告)
- [可变参数 va_list](#可变参数-va_list)
- [一句话总结](#一句话总结)

---

## 分级日志：info / debug / trace

日志分四级，数值越大越详细：

```c
typedef enum {
    LOG_OFF   = 0,   /* 静默, 只在出错时输出 */
    LOG_INFO  = 1,   /* 阶段性摘要 (默认) */
    LOG_DEBUG = 2,   /* 每层的关键统计量 */
    LOG_TRACE = 3,   /* 所有中间张量、每步细节 */
} LogLevel;
```

| 命令行参数 | 等级 | 显示内容 | 适用场景 |
|-----------|------|---------|---------|
| （无） | info | 阶段摘要（加载完成、生成几个 token） | 日常使用 |
| `-v` | debug | 每层的范数、激活率、注意力分布 | 检查数值是否健康 |
| `-vv` | trace | 所有中间张量的前几个数 | 对照 PyTorch 找 bug |
| `--report` | （强制 trace） | trace 内容渲染成 HTML | 离线分析、分享 |

`run.c` 里在 `main()` 开头预扫描这些参数：

```c
for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-v") == 0)        trace_set_level(LOG_DEBUG);
    else if (strcmp(argv[i], "-vv") == 0)  trace_set_level(LOG_TRACE);
    else if (strcmp(argv[i], "--report") == 0) trace_set_report(1);
}
/* 报告模式自动升级到 trace, 否则内容太少 */
if (trace_get_report() && trace_get_level() < LOG_TRACE) {
    trace_set_level(LOG_TRACE);
}
```

### 便捷宏

`trace.h` 提供了一组宏，让日志调用一行就能写完：

```c
#define LOGI(...)   log_printf(LOG_INFO,  __VA_ARGS__)   /* info, 总显示 */
#define LOGD(...)   log_printf(LOG_DEBUG, __VA_ARGS__)   /* debug 级才显示 */
#define LOGT(...)   log_printf(LOG_TRACE, __VA_ARGS__)   /* trace 级才显示 */

#define LOGI_L(cat, ind, ...)  log_line(LOG_INFO,  (cat), (ind), __VA_ARGS__)
#define LOGD_L(cat, ind, ...)  log_line(LOG_DEBUG, (cat), (ind), __VA_ARGS__)
#define LOGT_L(cat, ind, ...)  log_line(LOG_TRACE, (cat), (ind), __VA_ARGS__)
```

- `LOGI` / `LOGD` / `LOGT`：简单一行，无缩进无颜色
- `LOGI_L` / `LOGD_L` / `LOGT_L`：带类别（决定颜色）和缩进（决定树形符号）

`log_printf` 内部判断等级：

```c
void log_printf(LogLevel level, const char *fmt, ...) {
    if (level > g_level) return;   /* 等级不够直接返回, 零开销 */
    ...
}
```

等级不够的日志**完全跳过**（连参数格式化都不做），所以生产环境开 info 级不会有性能损失。

---

## stderr vs stdout：日志和生成分开

**这是整个日志系统最重要的设计决定**：日志走 stderr，生成文本走 stdout。

```c
/* 日志 → stderr */
fprintf(stderr, "%s\n", buf);

/* 生成的 token → stdout (run.c 里) */
fwrite(buf, 1, blen, stdout);
```

为什么分开？这样重定向时可以独立控制：

```bash
# 只存生成文本, 日志扔掉
./run model.safetensors tokenizer.bin "你好" > out.txt

# 只存日志, 生成文本扔掉
./run model.safetensors tokenizer.bin "你好" -vv 2> trace.log

# 两个都存
./run model.safetensors tokenizer.bin "你好" -vv > out.txt 2> trace.log

# 日志直接显示在屏幕, 生成文本存文件
./run model.safetensors tokenizer.bin "你好" -v > out.txt
```

如果日志和生成都走 stdout，这些场景就全乱套了——文本文件里会混入一堆彩色日志。

**对照表**：

| 流 | 用途 | 内容 |
|----|------|------|
| **stdout** | 生成结果 | 模型生成的 token 文本（给人看 / 管道传递） |
| **stderr** | 日志 | 加载进度、每层统计、采样决策（给开发者看） |

> 这是 Unix 哲学的经典应用：「程序输出**结果数据**到 stdout，输出**诊断信息**到 stderr」。

---

## TTY 检测与 ANSI 颜色

### TTY 检测

日志带颜色，但**重定向到文件时不该带颜色**（否则文件里全是 `\033[31m` 这种乱码）。怎么知道当前是终端还是管道？

```c
static int g_stderr_is_tty = -1;   /* 缓存, 避免每次调用 isatty */

static int stderr_is_tty(void) {
    if (g_stderr_is_tty < 0)
        g_stderr_is_tty = isatty(fileno(stderr));
    return g_stderr_is_tty;
}
```

**术语**：

- TTY** = TeleTYpewriter（终端的古老叫法，沿用至今）。现在泛指交互式终端。
- isatty** = "is a tty?"，判断文件描述符是不是终端。返回 1 是终端，0 是管道/文件。
- fileno** = 从 `FILE*`（如 `stderr`）拿到底层的整数文件描述符（如 2）。

缓存 `g_stderr_is_tty` 是因为 `isatty` 要做系统调用，频繁调用有开销。stderr 是不是终端在一次运行里不会变，所以查一次就够了。

### ANSI 转义码

终端识别的特殊字符序列，以 `\033[`（ESC + `[`）开头，控制颜色、粗体等：

```c
#define C_RESET   "\033[0m"      /* 重置所有样式 */
#define C_BOLD    "\033[1m"      /* 加粗 */
#define C_DIM     "\033[2m"      /* 暗淡 */
#define C_CYAN    "\033[36m"     /* 青色 */
#define C_GRAY    "\033[90m"     /* 亮黑（灰） */
#define C_YELLOW  "\033[33m"     /* 黄色 */
#define C_GREEN   "\033[32m"     /* 绿色 */
#define C_MAGENTA "\033[35m"     /* 紫色 */
#define C_RED     "\033[31m"     /* 红色 */
```

**用法**：在要变色的文字前插入颜色码，文字后插入 `C_RESET`：

```c
fprintf(stderr, "%s%s%s", C_RED, "这条是红色", C_RESET);
```

终端显示效果：

```
\033[31m红色文字\033[0m  →  红色文字 (红)
```

### 按类别着色

日志按**类别（LogCategory）**着色，不同信息用不同颜色，一眼就能分辨：

```c
typedef enum {
    CAT_TOKENIZER,   /* 青色:  分词相关 */
    CAT_LAYER,       /* 灰色:  层级标题 */
    CAT_OP,          /* 灰色:  算子内部 */
    CAT_VALUE,       /* 黄色:  数值 */
    CAT_ATTENTION,   /* 绿色:  注意力 */
    CAT_SAMPLE,      /* 紫色:  采样 */
    CAT_KEY,         /* 红色加粗: 关键节点 (阶段切换等) */
    CAT_TEXT,        /* 普通 */
} LogCategory;
```

类别到颜色的映射在 `cat_color()` 函数里：

```c
static const char *cat_color(LogCategory cat) {
    switch (cat) {
        case CAT_TOKENIZER: return C_CYAN;
        case CAT_LAYER:     return C_DIM;
        case CAT_OP:        return C_DIM;
        case CAT_VALUE:     return C_YELLOW;
        case CAT_ATTENTION: return C_GREEN;
        case CAT_SAMPLE:    return C_MAGENTA;
        case CAT_KEY:       return C_RED C_BOLD;
        case CAT_TEXT:      return "";
        default:            return "";
    }
}
```

### 输出时的条件染色

```c
void log_line(LogLevel level, LogCategory cat, int indent, const char *fmt, ...) {
    ...
    if (stderr_is_tty()) {
        const char *col = cat_color(cat);
        fprintf(stderr, "%s%s%s%s\n", *col ? col : "", prefix, body,
                *col ? C_RESET : "");
    } else {
        fprintf(stderr, "%s%s\n", prefix, body);   /* 管道/文件: 不染色 */
    }
    ...
}
```

非 TTY 时完全不输出颜色码，保证重定向到文件的内容干净。

---

## 向量统计：范数、激活率

光 dump 原始数字没用——一个 896 维向量打印出来是天书。日志系统提供几个**摘要函数**，把向量压缩成一两个有意义的数字。

### L2 范数（vec_norm）

```c
float vec_norm(const float *v, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)v[i] * v[i];
    return (float)sqrt(s);
}
```

**通俗解释**：向量的「长度」。`√(v[0]² + v[1]² + ... + v[895]²)`。

**作用**：日志里用范数摘要一个向量的整体大小。比如「第 5 层输出范数 = 87.3」，比打印 896 个数有用得多。

调试时关键看**范数是否稳定**——第 5.2 节那张表就是用范数监控每一层，发现不归一化会从 0.86 爆炸到 690 亿。

> 注意用 `double` 累加再开方，避免 896 个 `float` 平方和精度损失。这是数值编程的常见技巧。

### 激活率（vec_active_ratio）

```c
float vec_active_ratio(const float *v, int n, float threshold) {
    int cnt = 0;
    for (int i = 0; i < n; i++) if (v[i] > threshold) cnt++;
    return (float)cnt / n;
}
```

**通俗解释**：SwiGLU 后有多少比例的通道「亮了」（超过阈值）。反映 MLP 的稀疏性。

**作用**：检查 MLP 是否健康。如果激活率接近 0（所有通道都关）或接近 1（所有通道都开），说明 MLP 出问题了。

```
健康 SwiGLU 输出 (4864 维):
  激活率 ≈ 0.3-0.7  (大约一半通道开)
  → 模型在选择性加工信息

异常:
  激活率 = 0.01      → 几乎全关, MLP 没在工作
  激活率 = 0.99      → 几乎全开, 门控失效
```

### 其他统计

`trace.c` 还提供：

```c
float vec_mean(const float *v, int n);   /* 均值 */
float vec_max(const float *v, int n);    /* 最大值 */
float vec_min(const float *v, int n);    /* 最小值 */
```

加上 `log_vec()` 可以打印向量的前 k 个元素（trace 级用，对照 PyTorch 时用）：

```c
void log_vec(LogLevel level, const char *name, const float *v, int n, int k);
/* 输出形如: x[0:4] = [-0.0250, 0.0060, -0.0100, 0.0130] */
```

### 一条 debug 日志长什么样

开启 `-v` 时，每一层会打印类似这样的摘要：

```
┌─ Layer 5/24
│  ├─ RMSNorm   范数=87.3   均值=0.002
│  ├─ Q proj    范数=42.1   max=3.8   min=-4.1
│  ├─ K proj    范数=18.5
│  ├─ Attention 分数=[0.31, 0.28, 0.41, ...]  max=0.62
│  └─ MLP       激活率=0.43  范数=12.7
```

光看这些数字就能判断模型是否健康：范数稳定（不爆炸不消失）、激活率合理（0.3-0.7）、注意力有区分度（不是均匀分布）。

---

## HTML 报告

终端日志滚得快，看不清。`report.c` 把同一批日志渲染成一个**自包含的 HTML 文件**，可以慢慢看、可以分享。

### 用法

```bash
./run model.safetensors tokenizer.bin "你好" --report
# 运行结束后生成 trace_report.html
```

`--report` 会自动升级日志到 trace 级（否则报告内容太少）：

```c
if (trace_get_report() && trace_get_level() < LOG_TRACE) {
    trace_set_level(LOG_TRACE);
}
```

### 报告特性

```c
/* CSS 样式表 (内联, 单文件自包含) */
static const char *CSS =
"body { background:#1e1e2e; color:#cdd6f4; font-family:'SF Mono',Menlo,Consolas,monospace; ... }\n"
".tok    { color:#94e2d5; }    /* 青色: 分词器 */\n"
".layer  { color:#a6adc8; }    /* 灰色: 层 */\n"
".val    { color:#f9e2af; }    /* 黄色: 数值 */\n"
".att    { color:#a6e3a1; }    /* 绿色: 注意力 */\n"
".key    { color:#f38ba8; font-weight:bold; }  /* 红色加粗: 关键 */\n"
"details { margin:2px 0; }\n"
"summary { cursor:pointer; color:#89b4fa; }\n";
```

报告特性：

- 深色主题**（Catppuccin Mocha 配色，护眼）
- 等宽字体**（`SF Mono` / Menlo / Consolas，对齐整齐）
- 24 层可折叠**（用 HTML `<details>` 标签，点击展开/收起）
- 按类别着色**（CSS class 和终端颜色一一对应）
- 自包含**（CSS 全部内联，不依赖外部文件，单 HTML 就能打开）

### 折叠逻辑

`report.c` 用一个简单规则识别「该开新折叠块」的行：

```c
if (strstr(r->text, "┌─ Layer") || strstr(r->text, "═══")) {
    if (in_details) fprintf(f, "</details>\n");
    fprintf(f, "<details open>\n<summary class=\"%s\">%s</summary>\n",
            cls, escaped);
    in_details = 1;
    continue;
}
```

凡是带 `┌─ Layer` 或 `═══`（阶段分隔符）的行都开一个 `<details>` 块，后续行都塞进去，直到遇到下一个分隔符。这样 24 层每一层都能独立折叠。

### HTML 转义

把日志文本写进 HTML 前要转义，否则 `<` `>` `&` 会被当 HTML 标签解析：

```c
static int html_escape(const char *src, char *out, int outsize) {
    for (int i = 0; src[i] && p < outsize - 8; i++) {
        char c = src[i];
        if (c == '<')      p += snprintf(out+p, outsize-p, "&lt;");
        else if (c == '>') p += snprintf(out+p, outsize-p, "&gt;");
        else if (c == '&') p += snprintf(out+p, outsize-p, "&amp;");
        else if (c == '"') p += snprintf(out+p, outsize-p, "&quot;");
        else out[p++] = c;
    }
}
```

### 内存收集机制

日志不会立刻写进 HTML——而是先收集到一个内存结构 `TraceLog`，运行结束时一次性渲染：

```c
typedef struct {
    LogLevel    level;
    LogCategory cat;
    int         indent;
    char       *text;      /* 已格式化的文本 (深拷贝) */
} TraceRecord;

typedef struct {
    TraceRecord *records;
    int          count;
    int          capacity;
} TraceLog;
```

`log_line` 每次输出同时调 `trace_append()` 把记录塞进 `g_log`：

```c
static void trace_append(LogLevel level, LogCategory cat, int indent,
                         const char *text) {
    if (!g_report) return;   /* 不生成报告就不收集, 省内存 */
    ...
    r->text = strdup(text ? text : "");   /* 深拷贝, 不能只存指针 */
}
```

> **省内存技巧**：不生成报告时（`g_report == 0`）完全不收集，避免 trace 级日志吃光内存。
> 生成模式下，`run.c` 在每轮 decode 前调用 `trace_clear()` 清空，**只保留最后一个 token 的轨迹**——否则几百个 token 的 trace 会把内存撑爆。

---

## 可变参数 va_list

`LOGI("a=%d b=%f", a, b)` 这种「参数数量可变」的函数，靠 C 的 `va_list` 机制实现。

### 工作流程

```c
void log_printf(LogLevel level, const char *fmt, ...) {
    if (level > g_level) return;
    char buf[2048];

    va_list ap;            /* ① 声明一个遍历器 */
    va_start(ap, fmt);     /* ② 用最后一个固定参数初始化 */
    vsnprintf(buf, sizeof(buf), fmt, ap);   /* ③ 用 va_list 版的 printf */
    va_end(ap);            /* ④ 结束遍历, 清理 */

    fprintf(stderr, "%s\n", buf);
}
```

### 关键术语

| 术语 | 含义 |
|------|------|
| **`...`** | C 函数声明的「可变参数」标记，必须在参数列表最后 |
| **`va_list`** | 遍历可变参数的状态变量（实际是个指针） |
| **`va_start(ap, last_fixed)`** | 初始化 `va_list`，第二个参数是**最后一个固定参数**的名字 |
| **`va_arg(ap, type)`** | 取下一个参数，指定类型（本项目用 `vsnprintf` 自动取） |
| **`va_end(ap)`** | 结束遍历，必须配对调用 |
| **`vsnprintf(buf, n, fmt, ap)`** | `printf` 的 va_list 版本，把可变参数格式化进 buf |

### 为什么需要 `va_list`

`printf` 系列函数本身就靠 `va_list` 实现。我们想做一个**自己的 printf 包装函数**（加上等级判断、颜色、收集到内存），就要用 `vsnprintf` 把已经收好的 `va_list` 喂给 printf 的内核。

```
用户调用: LOGI("a=%d", 42)
   │
   │  展开 → log_printf(LOG_INFO, "a=%d", 42)
   │
   ▼
log_printf 内部:
   va_list ap;
   va_start(ap, fmt);          ← 从 fmt 之后开始取参数
   vsnprintf(buf, 2048, fmt, ap);   ← vsnprintf 自动取到 42, 填进 %d
   va_end(ap);
   │
   ▼
buf = "a=42" → 写到 stderr + 收集到内存
```

### 为什么不能直接 `snprintf(buf, n, fmt, ...)`

因为 `...` 不是真正的参数——你没法把它**转发**给另一个函数。`log_printf` 拿到的可变参数，必须先用 `va_list` 打包，再用 `vsnprintf` 处理。这是 C 语言可变参数函数的标准模式。

---

## 一句话总结

**日志系统 = 分级（info/debug/trace）+ 分流（stderr 日志 / stdout 生成）+ 摘要（范数/激活率把向量压成数字）+ 可视化（HTML 报告）**。
四件法宝配合，让一个 0.5B 模型的内部计算从黑盒变成可观测的过程——这是第 9 章调试方法论能成立的前提。
