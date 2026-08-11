# 最简推理引擎 — 纯 C，只依赖 libc
#
# 用法:
#   make            # 编译
#   make run        # 编译并跑一个示例 prompt
#   make info       # 打印 safetensors 里所有张量(调试用)
#   make clean

CC      = gcc
CFLAGS  = -O2 -std=c99 -Wall -Wno-unused-function -Wno-unused-result
LDFLAGS = -lm

# 源文件: 所有 .c 在同目录下
SRCS = run.c net.c safetensors.c json.c tokenizer.c trace.c report.c
OBJS = $(SRCS:.c=.o)

# 默认编译可执行文件 run
run: $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

# 通用编译规则
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# 头文件依赖: 任一 .h 改动就重新编译所有 .o
$(OBJS): net.h safetensors.h json.h tokenizer.h trace.h report.h

# 便捷目标
info: run
	./run info model.safetensors

.PHONY: clean
clean:
	rm -f $(OBJS) run
