#!/usr/bin/env python3
"""Extract the gpt2-BPE tokenizer (vocab + merges + special ids) from the north-mini-code GGUF.

Run (root, to read the ollama-owned blob):
    sudo python3 tools/extract_tokenizer.py <blob> <out_dir>

Writes out_dir/{vocab.json, merges.json, meta.json} (user-readable) so the tokenizer can be built without sudo."""
import json
import os
import struct
import sys

f = open(sys.argv[1], "rb")
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)


def rd(fmt): return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
def rstr(): return f.read(rd("Q")).decode("utf-8", "replace")
FIXED = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4), 5: ("i", 4),
         6: ("f", 4), 7: ("B", 1), 10: ("Q", 8), 11: ("q", 8), 12: ("d", 8)}


def rval(t):
    if t == 8:
        return rstr()
    if t == 9:
        et = rd("I"); n = rd("Q")
        if et == 8:
            return [rstr() for _ in range(n)]                      # materialize string arrays (tokens/merges)
        fmt, sz = FIXED[et]
        return list(struct.unpack(f"<{n}{fmt}", f.read(n * sz)))
    fmt, _ = FIXED[t]
    return rd(fmt)


assert f.read(4) == b"GGUF", "not a GGUF file"
ver = rd("I"); ntensor = rd("Q"); nkv = rd("Q")
kv = {}
for _ in range(nkv):
    k = rstr(); kv[k] = rval(rd("I"))

tokens = kv["tokenizer.ggml.tokens"]
merges = kv.get("tokenizer.ggml.merges", [])
meta = {
    "model": kv.get("tokenizer.ggml.model"),
    "pre": kv.get("tokenizer.ggml.pre"),
    "bos_token_id": kv.get("tokenizer.ggml.bos_token_id"),
    "eos_token_id": kv.get("tokenizer.ggml.eos_token_id"),
    "pad_token_id": kv.get("tokenizer.ggml.padding_token_id"),
    "unk_token_id": kv.get("tokenizer.ggml.unknown_token_id"),
    "add_bos_token": kv.get("tokenizer.ggml.add_bos_token"),
    "add_eos_token": kv.get("tokenizer.ggml.add_eos_token"),
    "n_vocab": len(tokens),
    "n_merges": len(merges),
    "token_type": kv.get("tokenizer.ggml.token_type"),
}
json.dump(tokens, open(os.path.join(OUT, "vocab.json"), "w"), ensure_ascii=False)
json.dump(merges, open(os.path.join(OUT, "merges.json"), "w"), ensure_ascii=False)
json.dump({k: v for k, v in meta.items() if k != "token_type"}, open(os.path.join(OUT, "meta.json"), "w"),
          ensure_ascii=False, indent=2)
json.dump(meta["token_type"], open(os.path.join(OUT, "token_type.json"), "w"))
for p in ("vocab.json", "merges.json", "meta.json", "token_type.json"):
    os.chmod(os.path.join(OUT, p), 0o644)
print(f"wrote {OUT}: vocab={len(tokens)} merges={len(merges)} "
      f"model={meta['model']} pre={meta['pre']} bos={meta['bos_token_id']} eos={meta['eos_token_id']}")
