/* report.c — 生成 HTML 可视化报告
 *
 * 把 TraceLog 收集的记录渲染成 trace_report.html。
 * 特性:
 *   - 深色主题, 等宽字体
 *   - 按类别着色 (和终端一致)
 *   - 层级缩进
 *   - "┌─ Layer" 行可点击折叠 (用 <details>)
 *
 * 用法: 运行时加 --report, 结束后自动生成。
 */

#include "report.h"
#include "trace.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* HTML 转义: 把 < > & " 替换成实体, 写到 out。
 * 返回写入长度。out 至少需要 6x src 的空间保守。*/
static int html_escape(const char *src, char *out, int outsize) {
    int p = 0;
    for (int i = 0; src[i] && p < outsize - 8; i++) {
        char c = src[i];
        if (c == '<')      { p += snprintf(out+p, outsize-p, "&lt;"); }
        else if (c == '>') { p += snprintf(out+p, outsize-p, "&gt;"); }
        else if (c == '&') { p += snprintf(out+p, outsize-p, "&amp;"); }
        else if (c == '"') { p += snprintf(out+p, outsize-p, "&quot;"); }
        else { out[p++] = c; }
    }
    out[p] = 0;
    return p;
}

/* 类别 → CSS 颜色 (和终端对应) */
static const char *cat_css_class(LogCategory cat) {
    switch (cat) {
        case CAT_TOKENIZER: return "tok";
        case CAT_LAYER:     return "layer";
        case CAT_OP:        return "op";
        case CAT_VALUE:     return "val";
        case CAT_ATTENTION: return "att";
        case CAT_SAMPLE:    return "sample";
        case CAT_KEY:       return "key";
        default:            return "txt";
    }
}

/* 缩进 → 空格数 */
static int indent_spaces(int indent) {
    return indent * 2;
}

/* CSS 样式表 (内联, 单文件自包含) */
static const char *CSS =
"body { background:#1e1e2e; color:#cdd6f4; font-family:'SF Mono',Menlo,Consolas,monospace;"
" font-size:13px; line-height:1.5; margin:0; padding:20px; }\n"
"h1 { color:#89b4fa; border-bottom:1px solid #45475a; padding-bottom:8px; }\n"
".summary { background:#313244; padding:12px 16px; border-radius:8px; margin-bottom:16px;"
" color:#a6adc8; }\n"
".log { white-space:pre-wrap; word-break:break-all; }\n"
".line { padding:1px 0; border-radius:2px; }\n"
".tok    { color:#94e2d5; }    /* 青色: 分词器 */\n"
".layer  { color:#a6adc8; }    /* 灰色: 层 */\n"
".op     { color:#9399b2; }    /* 灰色: 算子 */\n"
".val    { color:#f9e2af; }    /* 黄色: 数值 */\n"
".att    { color:#a6e3a1; }    /* 绿色: 注意力 */\n"
".sample { color:#cba6f7; }    /* 紫色: 采样 */\n"
".key    { color:#f38ba8; font-weight:bold; }  /* 红色加粗: 关键 */\n"
".txt    { color:#cdd6f4; }\n"
"details { margin:2px 0; }\n"
"summary { cursor:pointer; color:#89b4fa; padding:2px 0; }\n"
"summary:hover { color:#b4befe; }\n"
".footer { margin-top:20px; color:#585b70; font-size:11px; border-top:1px solid #45475a;"
" padding-top:8px; }\n"
;

int report_generate(const char *path) {
    TraceLog *tl = trace_get_log();
    if (!tl || tl->count == 0) {
        fprintf(stderr, "[报告] 没有收集到任何记录 (需要 --report 且日志级别足够)\n");
        return -1;
    }

    FILE *f = fopen(path, "w");
    if (!f) {
        perror("写 HTML 报告");
        return -1;
    }

    /* HTML 头部 */
    fprintf(f,
        "<!DOCTYPE html>\n<html lang=\"zh\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        "<title>推理引擎 Trace 报告</title>\n<style>\n%s</style>\n</head>\n<body>\n",
        CSS);

    fprintf(f, "<h1>推理引擎执行轨迹</h1>\n");
    fprintf(f, "<div class=\"summary\">共 %d 条记录。这是生成最后一个 token 时的完整前向传播细节。"
               "<br>点击带 ▶ 的行可展开/折叠。</div>\n",
            tl->count);

    fprintf(f, "<div class=\"log\">\n");

    int in_details = 0;   /* 是否在一个可折叠块里 */

    /* 逐条渲染 */
    for (int i = 0; i < tl->count; i++) {
        TraceRecord *r = &tl->records[i];
        char escaped[4096];
        html_escape(r->text, escaped, sizeof(escaped));

        const char *cls = cat_css_class(r->cat);
        int sp = indent_spaces(r->indent);

        /* "┌─ Layer" 开头的行: 开始一个可折叠 <details> */
        if (strstr(r->text, "┌─ Layer") || strstr(r->text, "═══")) {
            if (in_details) {
                fprintf(f, "</details>\n");
            }
            fprintf(f, "<details%s>\n<summary class=\"%s\">%s</summary>\n",
                    " open", cls, escaped);   /* 默认展开 */
            in_details = 1;
            continue;
        }

        /* 普通行: 缩进 + 颜色 */
        fprintf(f, "<div class=\"line %s\">", cls);
        for (int s = 0; s < sp; s++) fputc(' ', f);
        fprintf(f, "%s</div>\n", escaped);
    }

    if (in_details) {
        fprintf(f, "</details>\n");
    }

    fprintf(f, "</div>\n");   /* .log */

    fprintf(f, "<div class=\"footer\">由 inference 引擎的 trace 系统生成。"
               "用 -v / -vv 控制详细度。</div>\n");
    fprintf(f, "</body>\n</html>\n");

    fclose(f);
    return 0;
}
