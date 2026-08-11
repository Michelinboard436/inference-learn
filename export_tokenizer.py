#!/usr/bin/env python3
"""把 Qwen 的 tokenizer.json 导出成紧凑二进制 tokenizer.bin 给 C 引擎用。

用法:
    .venv/bin/python export_tokenizer.py

输出: tokenizer.bin

二进制布局 (全小端 uint32):
    [header: 4 个 uint32]
        vocab_size        N    (base vocab 条数, 151643)
        n_merges          M    (151387)
        n_special         S    (22)
        max_token_len     L    (最长 token 的字节数, 供 C 分配缓冲)
    [N 条 vocab 字符串]  每条: uint32 len + len 字节
    [M 条 merge]        每条: uint32 a + uint32 b   (vocab 字符串的 id)
    [S 条 special]      每条: uint32 id + uint32 len + len 字节
    [256 个 byte→unicode 映射]  每条: uint32 len + len 字节
        byte_map[i] = 第 i 个字节对应的 unicode "字符"的 UTF-8 编码
        (GPT-2 byte-level: 不可打印字节映射到 0x100-0x120 区间的 unicode)

注意: vocab 字符串里存放的是 byte-level 映射后的"字符"的 UTF-8 编码。
      例如空格 ' ' 在 byte-level 下变成 'Ġ' (U+0120), UTF-8 是 0xC4 0xA0。
      decode 时需要把这些字符再还原回原始字节。
"""
import json
import struct
import sys

# GPT-2 的 byte-level 映射 (和 tokenizers/huggingface 完全一致)
def bytes_to_unicode():
    """每个字节(0..255)映射到一个 unicode 字符。
    可打印的 ASCII/拉丁补充字符直接用原码点，其余映射到 256+ 区间。"""
    bs = (
        list(range(ord("!"), ord("~") + 1))            # 33-126
        + list(range(ord("¡"), ord("¬") + 1))           # 161-172
        + list(range(ord("®"), ord("ÿ") + 1))           # 174-255
    )
    cs = bs[:]                                          # 可见字符: 字节值 == 码点
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)                               # 不可见字节追加到末尾
            cs.append(256 + n)                         # 映射到 256+ 的可见区间
            n += 1
    # ★ 关键: 用原始字节值 bs 做 key (不是 range(256))
    return dict(zip(bs, [chr(c) for c in cs]))


def main():
    with open("tokenizer.json") as f:
        tj = json.load(f)

    model = tj["model"]
    vocab = model["vocab"]          # dict: str -> id
    merges_raw = model["merges"]    # list[str] "a b"
    added = tj["added_tokens"]      # list[dict]

    # 用 vocab 给每个 merge 的两边查 id
    vocab_size = len(vocab)
    merges = []
    for m in merges_raw:
        a, b = m.split(" ", 1)      # 只按第一个空格切
        if a not in vocab or b not in vocab:
            # 罕见: merge 的部分不在 vocab 里 (理论上不该发生)
            continue
        merges.append((vocab[a], vocab[b]))
    n_merges = len(merges)

    # special tokens (按 id 排序)
    added.sort(key=lambda x: x["id"])
    specials = [(a["id"], a["content"]) for a in added]
    n_special = len(specials)

    # vocab 按 id 排序成列表，存 UTF-8 字节
    vocab_by_id = sorted(vocab.items(), key=lambda kv: kv[1])
    vocab_list = [v[0] for v in vocab_by_id]
    max_token_len = max(len(s.encode("utf-8")) for s in vocab_list)

    # byte map
    bmap = bytes_to_unicode()       # int -> str(1 char)

    # 写二进制
    out = bytearray()
    out += struct.pack("<IIII", vocab_size, n_merges, n_special, max_token_len)

    # vocab
    for s in vocab_list:
        b = s.encode("utf-8")
        out += struct.pack("<I", len(b))
        out += b

    # merges
    for a, b in merges:
        out += struct.pack("<II", a, b)

    # specials
    for sid, content in specials:
        b = content.encode("utf-8")
        out += struct.pack("<II", sid, len(b))
        out += b

    # byte map: 256 条
    for i in range(256):
        b = bmap[i].encode("utf-8")
        out += struct.pack("<I", len(b))
        out += b

    with open("tokenizer.bin", "wb") as f:
        f.write(out)

    print(f"已写出 tokenizer.bin ({len(out)} 字节)")
    print(f"  vocab_size = {vocab_size}")
    print(f"  n_merges   = {n_merges}")
    print(f"  n_special  = {n_special}")
    print(f"  max_token_len = {max_token_len}")
    print(f"\n用法: ./run model.safetensors tokenizer.bin \"你的提示\"")


if __name__ == "__main__":
    main()
