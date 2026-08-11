#ifndef TRACE_H
#define TRACE_H

/*
 * trace.h — 分级日志系统
 *
 * 用法:
 *   -v     开启 debug 级 (每层摘要)
 *   -vv    开启 trace 级 (所有中间张量)
 *   --report 运行结束后生成 trace_report.html
 *
 * 日志全部走 stderr, 不污染 stdout 的生成文本。
 * 所有日志同时收集到内存 TraceLog, 供 report.c 生成 HTML。
 *
 * 典型用法:
 *   trace_set_level(LOG_DEBUG);
 *   LOGI("加载完成");                          // info, 总是显示
 *   LOGD("层 %d 范数=%.3f", l, norm);          // debug 级才显示
 *   LOGT("q[0:4] = %.4f %.4f ...", ...);       // trace 级才显示
 */

#include <stddef.h>

/* 日志等级 (数值越大越详细) */
typedef enum {
    LOG_OFF   = 0,   /* 静默, 只在出错时输出 */
    LOG_INFO  = 1,   /* 阶段性摘要 (默认) */
    LOG_DEBUG = 2,   /* 每层的关键统计量 */
    LOG_TRACE = 3,   /* 所有中间张量、每步细节 */
} LogLevel;

/* ============ 全局控制 ============ */

/* 设置当前日志等级。默认 LOG_INFO。*/
void trace_set_level(LogLevel level);
LogLevel trace_get_level(void);

/* 是否生成 HTML 报告 */
void trace_set_report(int on);
int  trace_get_report(void);

/* ============ 日志输出 ============ */

/* 带等级的 printf。level <= g_level 时才输出到 stderr。
 * 不带颜色/缩进, 适合一行式摘要。 */
void log_printf(LogLevel level, const char *fmt, ...);

/* 便捷宏。
 * 简单一行(无缩进):  LOGI("msg"), LOGD("msg"), LOGT("msg")
 * 带类别和缩进:       LOGD_L(cat, indent, "msg"), LOGT_L(cat, indent, "msg")
 */
#define LOGI(...)   log_printf(LOG_INFO,  __VA_ARGS__)
#define LOGD(...)   log_printf(LOG_DEBUG, __VA_ARGS__)
#define LOGT(...)   log_printf(LOG_TRACE, __VA_ARGS__)
#define LOGI_L(cat, ind, ...)   log_line(LOG_INFO,  (cat), (ind), __VA_ARGS__)
#define LOGD_L(cat, ind, ...)   log_line(LOG_DEBUG, (cat), (ind), __VA_ARGS__)
#define LOGT_L(cat, ind, ...)   log_line(LOG_TRACE, (cat), (ind), __VA_ARGS__)

/* ============ 结构化输出 (带缩进和树形符号) ============ */

/* 日志类别, 决定颜色 */
typedef enum {
    CAT_TOKENIZER,   /* 青色 */
    CAT_LAYER,       /* 灰色 */
    CAT_OP,          /* 灰色, 缩进 */
    CAT_VALUE,       /* 黄色 (数值) */
    CAT_ATTENTION,   /* 绿色 */
    CAT_SAMPLE,      /* 紫色 */
    CAT_KEY,         /* 红色加粗 (关键节点) */
    CAT_TEXT,        /* 普通 */
} LogCategory;

/* 输出一行带类别颜色的日志。
 * indent: 缩进层级 (0=顶层, 1=层内, 2=算子内)
 * 自动加树形符号 (├─ └─ │) 和颜色。*/
void log_line(LogLevel level, LogCategory cat, int indent,
              const char *fmt, ...);

/* ============ 向量统计工具 (观测点常用) ============ */

/* 算 L2 范数 */
float vec_norm(const float *v, int n);
/* 算均值 */
float vec_mean(const float *v, int n);
/* 算最大值 */
float vec_max(const float *v, int n);
/* 算最小值 */
float vec_min(const float *v, int n);
/* 算非零(>阈值)的比例, 用于 MLP 激活率 */
float vec_active_ratio(const float *v, int n, float threshold);

/* 打印向量的前 k 个元素 (用于 trace 级 dump) */
void log_vec(LogLevel level, const char *name, const float *v, int n, int k);

/* ============ HTML 报告收集 ============ */

/* 一条结构化记录, 供 HTML 报告使用 */
typedef struct {
    LogLevel    level;
    LogCategory cat;
    int         indent;
    char       *text;      /* 已格式化的文本 */
} TraceRecord;

/* 全局记录缓冲。log_line 会自动往这里追加。*/
typedef struct {
    TraceRecord *records;
    int          count;
    int          capacity;
} TraceLog;

/* 获取全局 TraceLog (单例) */
TraceLog *trace_get_log(void);

/* 清空收集的记录 (生成新一轮前向前调用) */
void trace_clear(void);

/* 释放记录缓冲内存 */
void trace_free(void);

#endif // TRACE_H
