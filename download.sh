#!/usr/bin/env bash
# 用 curl 从 hf-mirror.com 下载 Qwen2.5-0.5B
# (新版 hf CLI 强制校验 huggingface.co，国内直连超时，所以改用 curl)
#
# 用法:
#   ./download.sh                 # 用 hf-mirror 镜像
#   ./download.sh --hf            # 直连 HuggingFace(需网络通畅)
#
# 下载内容:
#   model.safetensors   ~988 MB   权重(bf16)
#   config.json         ~700 B    架构参数
#   tokenizer.json      ~7 MB     分词器

set -e

MODEL="Qwen/Qwen2.5-0.5B"

if [ "$1" = "--hf" ]; then
    BASE="https://huggingface.co"
else
    BASE="https://hf-mirror.com"
fi

FILES="config.json tokenizer.json model.safetensors"

echo "==> 从 $BASE 下载 $MODEL"
echo

for f in $FILES; do
    if [ -f "$f" ]; then
        echo "  [skip] $f 已存在"
        continue
    fi
    echo "  [get]  $f ..."
    # -L 跟随重定向; --fail 在 HTTP 错误时返回非零
    curl -L --fail --progress-bar \
         -o "$f" \
         "$BASE/$MODEL/resolve/main/$f"
    echo "  [ok]   $f  ($(ls -lh "$f" | awk '{print $5}'))"
done

echo
echo "==> 完成:"
ls -lh $FILES
