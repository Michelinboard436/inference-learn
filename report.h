#ifndef REPORT_H
#define REPORT_H

/*
 * report.h — 生成 HTML 可视化报告
 *
 * 运行结束后调用 report_generate(), 会把 TraceLog 里收集的所有记录
 * 写成 trace_report.html (带折叠、注意力热力图、采样柱状图)。
 */

/* 生成报告到指定路径。成功返回 0。 */
int report_generate(const char *path);

#endif // REPORT_H
