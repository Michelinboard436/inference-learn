/* trace.c — 分级日志系统实现
 *
 * 输出策略:
 *   - log_printf / log_line: 输出到 stderr (彩色), 同时收集到内存
 *   - 向量统计工具: 给观测点算范数/均值等, 一行代码就能输出有用的摘要
 *
 * 颜色用 ANSI 转义码, 非 TTY 时自动关闭 (管道/重定向不染色)。
 */

#include "trace.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

/* ============ 全局状态 ============ */

static LogLevel g_level = LOG_INFO;
static int      g_report = 0;

/* TTY 检测缓存 (避免每次调用 isatty) */
static int g_stderr_is_tty = -1;

void trace_set_level(LogLevel level) { g_level = level; }
LogLevel trace_get_level(void)       { return g_level; }
void trace_set_report(int on)        { g_report = on; }
int  trace_get_report(void)          { return g_report; }

/* ============ TraceLog 内存收集 ============ */

static TraceLog g_log = { NULL, 0, 0 };

TraceLog *trace_get_log(void) { return &g_log; }

/* 往 g_log 追加一条记录 (深拷贝 text) */
static void trace_append(LogLevel level, LogCategory cat, int indent,
                         const char *text) {
    if (!g_report) return;   /* 不生成报告就不收集, 省内存 */
    if (g_log.count >= g_log.capacity) {
        int newcap = g_log.capacity ? g_log.capacity * 2 : 1024;
        TraceRecord *p = realloc(g_log.records, newcap * sizeof(TraceRecord));
        if (!p) return;   /* 内存不够就丢弃, 不影响运行 */
        g_log.records = p;
        g_log.capacity = newcap;
    }
    TraceRecord *r = &g_log.records[g_log.count++];
    r->level = level;
    r->cat = cat;
    r->indent = indent;
    r->text = strdup(text ? text : "");
}

void trace_clear(void) {
    if (g_log.records) {
        for (int i = 0; i < g_log.count; i++) free(g_log.records[i].text);
        g_log.count = 0;
    }
}

void trace_free(void) {
    trace_clear();
    free(g_log.records);
    g_log.records = NULL;
    g_log.capacity = 0;
}

/* ============ 颜色 ============ */

/* 检查 stderr 是不是真终端。管道/重定向时不染色。*/
static int stderr_is_tty(void) {
    if (g_stderr_is_tty < 0)
        g_stderr_is_tty = isatty(fileno(stderr));
    return g_stderr_is_tty;
}

/* ANSI 颜色码 */
#define C_RESET   "\033[0m"
#define C_BOLD    "\033[1m"
#define C_DIM     "\033[2m"
/* 类别 → 颜色 */
#define C_CYAN    "\033[36m"
#define C_GRAY    "\033[90m"
#define C_YELLOW  "\033[33m"
#define C_GREEN   "\033[32m"
#define C_MAGENTA "\033[35m"
#define C_RED     "\033[31m"

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

/* 生成缩进前缀。indent=0 无前缀; 1 = "  ├─ "; 2 = "  │  ├─ " 等。
 * 这里用简化版: 每层 2 空格 + 第 1 层起加 "├─" 符号。*/
static void make_indent_prefix(int indent, char *buf, int bufsize) {
    if (indent <= 0) { buf[0] = 0; return; }
    int p = 0;
    for (int i = 0; i < indent - 1 && p < bufsize - 4; i++) {
        buf[p++] = ' '; buf[p++] = ' ';
    }
    /* 最后一层用 ├─ */
    if (p < bufsize - 4) {
        buf[p++] = ' '; buf[p++] = ' ';
    }
    buf[p] = 0;
}

/* ============ 输出函数 ============ */

void log_printf(LogLevel level, const char *fmt, ...) {
    if (level > g_level) return;
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    fprintf(stderr, "%s\n", buf);

    /* 收集到内存 (报告用), 作为无类别的纯文本 */
    trace_append(level, CAT_TEXT, 0, buf);
}

void log_line(LogLevel level, LogCategory cat, int indent,
              const char *fmt, ...) {
    if (level > g_level) return;

    char body[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(body, sizeof(body), fmt, ap);
    va_end(ap);

    char prefix[64];
    make_indent_prefix(indent, prefix, sizeof(prefix));

    /* stderr 彩色输出 */
    if (stderr_is_tty()) {
        const char *col = cat_color(cat);
        fprintf(stderr, "%s%s%s%s%s\n",
                *col ? col : "",
                prefix, body,
                *col ? C_RESET : "",
                "");
    } else {
        fprintf(stderr, "%s%s\n", prefix, body);
    }

    /* 收集 */
    char full[2200];
    snprintf(full, sizeof(full), "%s%s", prefix, body);
    trace_append(level, cat, indent, full);
}

/* ============ 向量统计 ============ */

float vec_norm(const float *v, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)v[i] * v[i];
    return (float)sqrt(s);
}

float vec_mean(const float *v, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += v[i];
    return (float)(s / n);
}

float vec_max(const float *v, int n) {
    float m = v[0];
    for (int i = 1; i < n; i++) if (v[i] > m) m = v[i];
    return m;
}

float vec_min(const float *v, int n) {
    float m = v[0];
    for (int i = 1; i < n; i++) if (v[i] < m) m = v[i];
    return m;
}

float vec_active_ratio(const float *v, int n, float threshold) {
    int cnt = 0;
    for (int i = 0; i < n; i++) if (v[i] > threshold) cnt++;
    return (float)cnt / n;
}

/* 打印向量的前 k 个元素 */
void log_vec(LogLevel level, const char *name, const float *v, int n, int k) {
    if (level > g_level) return;
    if (k > n) k = n;
    char buf[1024];
    int p = snprintf(buf, sizeof(buf), "%s[0:%d] = [", name, k);
    for (int i = 0; i < k && p < (int)sizeof(buf) - 32; i++) {
        p += snprintf(buf + p, sizeof(buf) - p, "%.4f%s", v[i], i+1<k ? ", " : "");
    }
    p += snprintf(buf + p, sizeof(buf) - p, "]");
    log_line(level, CAT_VALUE, 2, "%s", buf);
}
