# Chapter 8 · Logging & Visualization (trace.c / report.c)

> 🌐 [中文版](../08-logging) | English

> Chapter 8 | [Previous: Chapter 7 · Sampling & Generation](./07-sampling.md) | [Next: Chapter 9 · Debugging & Verification Methodology](./09-debugging.md)

> Corresponding source: `trace.h` / `trace.c` (~210 lines) + `report.h` / `report.c` (~145 lines)
> This chapter answers one question: **once the engine is running, how do you see what happens at every internal step.**

The most frustrating thing about an inference engine isn't "it won't run" — it's "it runs but the output is garbage, and you don't know which step went wrong."
The logging and visualization system covered in this chapter lets you **see** the actual numbers behind every matrix multiply, every normalization, every attention step.

---

## Table of Contents

- [Tiered Logging: info / debug / trace](#tiered-logging-info--debug--trace)
- [stderr vs stdout: Separate Logs from Generation](#stderr-vs-stdout-separate-logs-from-generation)
- [TTY Detection and ANSI Colors](#tty-detection-and-ansi-colors)
- [Vector Statistics: Norm, Activation Ratio](#vector-statistics-norm-activation-ratio)
- [HTML Report](#html-report)
- [Variadic Functions: va_list](#variadic-functions-va_list)
- [One-Sentence Summary](#one-sentence-summary)

---

## Tiered Logging: info / debug / trace

Logging is divided into four levels; the larger the number, the more verbose:

```c
typedef enum {
    LOG_OFF   = 0,   /* silent, only output on error */
    LOG_INFO  = 1,   /* stage-level summary (default) */
    LOG_DEBUG = 2,   /* key statistics for each layer */
    LOG_TRACE = 3,   /* all intermediate tensors, every step in detail */
} LogLevel;
```

| CLI argument | Level | What it shows | Use case |
|-----------|------|---------|---------|
| (none) | info | Stage summaries (load complete, how many tokens generated) | Daily use |
| `-v` | debug | Per-layer norm, activation ratio, attention distribution | Checking numerical health |
| `-vv` | trace | First few numbers of every intermediate tensor | Comparing against PyTorch to find bugs |
| `--report` | (forced trace) | Trace content rendered as HTML | Offline analysis, sharing |

In `run.c`, `main()` pre-scans these arguments at the start:

```c
for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-v") == 0)        trace_set_level(LOG_DEBUG);
    else if (strcmp(argv[i], "-vv") == 0)  trace_set_level(LOG_TRACE);
    else if (strcmp(argv[i], "--report") == 0) trace_set_report(1);
}
/* Report mode auto-upgrades to trace, otherwise there's too little content */
if (trace_get_report() && trace_get_level() < LOG_TRACE) {
    trace_set_level(LOG_TRACE);
}
```

### Convenience Macros

`trace.h` provides a set of macros so a log call fits on a single line:

```c
#define LOGI(...)   log_printf(LOG_INFO,  __VA_ARGS__)   /* info, always shown */
#define LOGD(...)   log_printf(LOG_DEBUG, __VA_ARGS__)   /* only shown at debug level */
#define LOGT(...)   log_printf(LOG_TRACE, __VA_ARGS__)   /* only shown at trace level */

#define LOGI_L(cat, ind, ...)  log_line(LOG_INFO,  (cat), (ind), __VA_ARGS__)
#define LOGD_L(cat, ind, ...)  log_line(LOG_DEBUG, (cat), (ind), __VA_ARGS__)
#define LOGT_L(cat, ind, ...)  log_line(LOG_TRACE, (cat), (ind), __VA_ARGS__)
```

- `LOGI` / `LOGD` / `LOGT`: simple single line, no indentation, no color
- `LOGI_L` / `LOGD_L` / `LOGT_L`: with a category (determines color) and indentation (determines the tree symbol)

`log_printf` checks the level internally:

```c
void log_printf(LogLevel level, const char *fmt, ...) {
    if (level > g_level) return;   /* level too low, return immediately, zero overhead */
    ...
}
```

Logs below the current level are **completely skipped** (not even the arguments get formatted), so running at info level in production incurs no performance cost.

---

## stderr vs stdout: Separate Logs from Generation

**This is the single most important design decision in the whole logging system**: logs go to stderr, generated text goes to stdout.

```c
/* logs → stderr */
fprintf(stderr, "%s\n", buf);

/* generated tokens → stdout (in run.c) */
fwrite(buf, 1, blen, stdout);
```

Why separate them? So redirection can control each independently:

```bash
# Save only the generated text, discard logs
./run model.safetensors tokenizer.bin "hello" > out.txt

# Save only the logs, discard generated text
./run model.safetensors tokenizer.bin "hello" -vv 2> trace.log

# Save both
./run model.safetensors tokenizer.bin "hello" -vv > out.txt 2> trace.log

# Show logs on screen, save generated text to file
./run model.safetensors tokenizer.bin "hello" -v > out.txt
```

If both logs and generation went to stdout, all these scenarios would break — the text file would end up polluted with colored log lines.

**Comparison table**:

| Stream | Purpose | Content |
|----|------|------|
| **stdout** | Generation result | The token text the model produces (for humans / piping) |
| **stderr** | Logs | Load progress, per-layer statistics, sampling decisions (for developers) |

> This is a classic application of the Unix philosophy: "programs write **result data** to stdout, and **diagnostic information** to stderr."

---

## TTY Detection and ANSI Colors

### TTY Detection

Logs are colored, but **when redirected to a file they should not carry color** (otherwise the file fills up with junk like `\033[31m`). How do you know whether the current target is a terminal or a pipe?

```c
static int g_stderr_is_tty = -1;   /* cached, to avoid calling isatty every time */

static int stderr_is_tty(void) {
    if (g_stderr_is_tty < 0)
        g_stderr_is_tty = isatty(fileno(stderr));
    return g_stderr_is_tty;
}
```

**Terminology**:

- TTY** = TeleTYpewriter (an old name for a terminal, still in use). Now broadly means an interactive terminal.
- isatty** = "is a tty?", checks whether a file descriptor is a terminal. Returns 1 for a terminal, 0 for a pipe/file.
- fileno** = extracts the underlying integer file descriptor (e.g. 2) from a `FILE*` (such as `stderr`).

`g_stderr_is_tty` is cached because `isatty` performs a system call, and frequent calls have overhead. Whether stderr is a terminal won't change during a single run, so checking once is enough.

### ANSI Escape Codes

Special character sequences that terminals recognize. They start with `\033[` (ESC + `[`) and control color, bold, and more:

```c
#define C_RESET   "\033[0m"      /* reset all styles */
#define C_BOLD    "\033[1m"      /* bold */
#define C_DIM     "\033[2m"      /* dim */
#define C_CYAN    "\033[36m"     /* cyan */
#define C_GRAY    "\033[90m"     /* bright black (gray) */
#define C_YELLOW  "\033[33m"     /* yellow */
#define C_GREEN   "\033[32m"     /* green */
#define C_MAGENTA "\033[35m"     /* magenta */
#define C_RED     "\033[31m"     /* red */
```

**Usage**: insert the color code before the text you want colored, and `C_RESET` after it:

```c
fprintf(stderr, "%s%s%s", C_RED, "this line is red", C_RESET);
```

How the terminal renders it:

```
\033[31mred text\033[0m  →  red text (in red)
```

### Coloring by Category

Logs are colored by **category (LogCategory)** — different kinds of information get different colors so you can tell them apart at a glance:

```c
typedef enum {
    CAT_TOKENIZER,   /* cyan:    tokenizer-related */
    CAT_LAYER,       /* gray:    layer-level headers */
    CAT_OP,          /* gray:    inside an operator */
    CAT_VALUE,       /* yellow:  numerical values */
    CAT_ATTENTION,   /* green:   attention */
    CAT_SAMPLE,      /* magenta: sampling */
    CAT_KEY,         /* red bold: key milestones (stage switches, etc.) */
    CAT_TEXT,        /* plain */
} LogCategory;
```

The category-to-color mapping lives in the `cat_color()` function:

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

### Conditional Coloring on Output

```c
void log_line(LogLevel level, LogCategory cat, int indent, const char *fmt, ...) {
    ...
    if (stderr_is_tty()) {
        const char *col = cat_color(cat);
        fprintf(stderr, "%s%s%s%s\n", *col ? col : "", prefix, body,
                *col ? C_RESET : "");
    } else {
        fprintf(stderr, "%s%s\n", prefix, body);   /* pipe/file: no color */
    }
    ...
}
```

When not a TTY, no color codes are emitted at all, keeping the redirected output clean.

---

## Vector Statistics: Norm, Activation Ratio

Dumping raw numbers is useless — printing an 896-dimensional vector is gibberish. The logging system provides several **summary functions** that compress a vector down to one or two meaningful numbers.

### L2 Norm (vec_norm)

```c
float vec_norm(const float *v, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)v[i] * v[i];
    return (float)sqrt(s);
}
```

**In plain terms**: the "length" of the vector. `√(v[0]² + v[1]² + ... + v[895]²)`.

**Purpose**: in logs, the norm summarizes the overall magnitude of a vector. For example, "Layer 5 output norm = 87.3" is far more useful than printing 896 numbers.

When debugging, the key thing to watch is **whether the norm stays stable** — the table in Section 5.2 uses the norm to monitor each layer and shows that without normalization the value explodes from 0.86 to 69 billion.

> Note the use of `double` accumulation before taking the square root, to avoid precision loss when summing 896 `float` squares. This is a common technique in numerical programming.

### Activation Ratio (vec_active_ratio)

```c
float vec_active_ratio(const float *v, int n, float threshold) {
    int cnt = 0;
    for (int i = 0; i < n; i++) if (v[i] > threshold) cnt++;
    return (float)cnt / n;
}
```

**In plain terms**: after SwiGLU, what fraction of channels are "lit" (above the threshold). Reflects the sparsity of the MLP.

**Purpose**: checks whether the MLP is healthy. If the activation ratio is near 0 (all channels off) or near 1 (all channels on), the MLP is malfunctioning.

```
Healthy SwiGLU output (4864-dim):
  activation ratio ≈ 0.3-0.7  (about half the channels on)
  → the model is selectively processing information

Abnormal:
  activation ratio = 0.01      → almost all off, MLP not working
  activation ratio = 0.99      → almost all on, gating has failed
```

### Other Statistics

`trace.c` also provides:

```c
float vec_mean(const float *v, int n);   /* mean */
float vec_max(const float *v, int n);    /* max */
float vec_min(const float *v, int n);    /* min */
```

Plus `log_vec()` can print the first k elements of a vector (used at trace level, for comparing against PyTorch):

```c
void log_vec(LogLevel level, const char *name, const float *v, int n, int k);
/* Output looks like: x[0:4] = [-0.0250, 0.0060, -0.0100, 0.0130] */
```

### What a debug log line looks like

With `-v` enabled, each layer prints a summary like this:

```
┌─ Layer 5/24
│  ├─ RMSNorm   norm=87.3   mean=0.002
│  ├─ Q proj    norm=42.1   max=3.8   min=-4.1
│  ├─ K proj    norm=18.5
│  ├─ Attention scores=[0.31, 0.28, 0.41, ...]  max=0.62
│  └─ MLP       activation_ratio=0.43  norm=12.7
```

Just looking at these numbers tells you whether the model is healthy: norms are stable (no explosion or vanishing), activation ratio is reasonable (0.3-0.7), attention has discriminating power (not a uniform distribution).

---

## HTML Report

Terminal logs scroll by fast and are hard to read. `report.c` renders the same set of logs into a **self-contained HTML file** that you can browse at your own pace and share.

### Usage

```bash
./run model.safetensors tokenizer.bin "hello" --report
# After the run finishes, trace_report.html is generated
```

`--report` automatically upgrades the log level to trace (otherwise the report would be too sparse):

```c
if (trace_get_report() && trace_get_level() < LOG_TRACE) {
    trace_set_level(LOG_TRACE);
}
```

### Report Features

```c
/* CSS stylesheet (inlined, single self-contained file) */
static const char *CSS =
"body { background:#1e1e2e; color:#cdd6f4; font-family:'SF Mono',Menlo,Consolas,monospace; ... }\n"
".tok    { color:#94e2d5; }    /* cyan: tokenizer */\n"
".layer  { color:#a6adc8; }    /* gray: layer */\n"
".val    { color:#f9e2af; }    /* yellow: values */\n"
".att    { color:#a6e3a1; }    /* green: attention */\n"
".key    { color:#f38ba8; font-weight:bold; }  /* red bold: key */\n"
"details { margin:2px 0; }\n"
"summary { cursor:pointer; color:#89b4fa; }\n";
```

Report features:

- Dark theme** (Catppuccin Mocha palette, easy on the eyes)
- Monospace font** (`SF Mono` / Menlo / Consolas, neat alignment)
- 24 collapsible layers** (using the HTML `<details>` tag, click to expand/collapse)
- Colored by category** (CSS classes correspond one-to-one with terminal colors)
- Self-contained** (all CSS inlined, no external files, a single HTML is enough to open)

### Collapsing Logic

`report.c` uses a simple rule to detect which lines "should start a new collapsible block":

```c
if (strstr(r->text, "┌─ Layer") || strstr(r->text, "═══")) {
    if (in_details) fprintf(f, "</details>\n");
    fprintf(f, "<details open>\n<summary class=\"%s\">%s</summary>\n",
            cls, escaped);
    in_details = 1;
    continue;
}
```

Any line containing `┌─ Layer` or `═══` (a stage separator) starts a new `<details>` block. Subsequent lines are appended into it until the next separator is hit. This way each of the 24 layers can be folded independently.

### HTML Escaping

Before writing log text into HTML, it must be escaped, otherwise `<`, `>`, `&` would be parsed as HTML tags:

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

### In-Memory Collection

Logs are not written into the HTML immediately — they are first collected into an in-memory structure `TraceLog`, then rendered all at once at the end of the run:

```c
typedef struct {
    LogLevel    level;
    LogCategory cat;
    int         indent;
    char       *text;      /* already-formatted text (deep copy) */
} TraceRecord;

typedef struct {
    TraceRecord *records;
    int          count;
    int          capacity;
} TraceLog;
```

Every time `log_line` outputs, it also calls `trace_append()` to push the record into `g_log`:

```c
static void trace_append(LogLevel level, LogCategory cat, int indent,
                         const char *text) {
    if (!g_report) return;   /* don't collect if no report is being generated, saves memory */
    ...
    r->text = strdup(text ? text : "");   /* deep copy, can't just store the pointer */
}
```

> **Memory-saving trick**: when not generating a report (`g_report == 0`), nothing is collected at all, avoiding trace-level logs eating up all memory.
> In report mode, `run.c` calls `trace_clear()` before each decode round to wipe the buffer, **keeping only the trace for the last token** — otherwise traces from hundreds of tokens would blow up memory.

---

## Variadic Functions: va_list

Functions like `LOGI("a=%d b=%f", a, b)` that take a "variable number of arguments" rely on C's `va_list` mechanism.

### How It Works

```c
void log_printf(LogLevel level, const char *fmt, ...) {
    if (level > g_level) return;
    char buf[2048];

    va_list ap;            /* 1. declare an iterator */
    va_start(ap, fmt);     /* 2. initialize using the last fixed parameter */
    vsnprintf(buf, sizeof(buf), fmt, ap);   /* 3. use the va_list version of printf */
    va_end(ap);            /* 4. finish iterating, cleanup */

    fprintf(stderr, "%s\n", buf);
}
```

### Key Terms

| Term | Meaning |
|------|------|
| **`...`** | The "variadic arguments" marker in a C function declaration, must be last in the parameter list |
| **`va_list`** | The state variable for iterating variadic arguments (effectively a pointer) |
| **`va_start(ap, last_fixed)`** | Initializes a `va_list`; the second argument is the name of the **last fixed parameter** |
| **`va_arg(ap, type)`** | Fetches the next argument with a specified type (this project uses `vsnprintf` to fetch automatically) |
| **`va_end(ap)`** | Ends iteration, must be called in a pair with `va_start` |
| **`vsnprintf(buf, n, fmt, ap)`** | The va_list version of `printf`, formats the variadic arguments into buf |

### Why `va_list` Is Needed

The `printf` family itself is implemented with `va_list`. If we want to write **our own printf wrapper** (adding level checks, colors, in-memory collection), we have to use `vsnprintf` to feed the already-collected `va_list` into printf's core.

```
User calls: LOGI("a=%d", 42)
   │
   │  expands to → log_printf(LOG_INFO, "a=%d", 42)
   │
   ▼
inside log_printf:
   va_list ap;
   va_start(ap, fmt);          ← start fetching arguments after fmt
   vsnprintf(buf, 2048, fmt, ap);   ← vsnprintf auto-fetches 42, fills into %d
   va_end(ap);
   │
   ▼
buf = "a=42" → written to stderr + collected into memory
```

### Why You Can't Just `snprintf(buf, n, fmt, ...)`

Because `...` is not a real parameter — you can't **forward** it to another function. The variadic arguments that `log_printf` receives must first be packed into a `va_list`, then handled by `vsnprintf`. This is the standard pattern for variadic functions in C.

---

## One-Sentence Summary

**The logging system = tiering (info/debug/trace) + stream separation (stderr for logs / stdout for generation) + summaries (norm/activation ratio compress vectors into numbers) + visualization (HTML report)**.
These four pieces working together turn the internal computation of a 0.5B model from a black box into an observable process — which is the prerequisite that makes the debugging methodology of Chapter 9 possible.
