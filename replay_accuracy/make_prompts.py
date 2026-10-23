#!/usr/bin/env python3
"""Build teacher-forced prompts from real text for the replay-accuracy runs.

Three sources, chosen so that most of the text post-dates the model's training data:
  code_py  the DeepInfer Python sources (released 2026-10-20), files in sorted order
  code_rs  the deepseek-recipe Rust sources (released 2026-10-20)
  prose    a set of long English blog posts

For each source and each prompt length L, a fresh, non-overlapping stretch of the
token stream is cut into L prompt tokens plus 256 golden continuation tokens, so no
two requests share a prefix (the engine's in-GPU prefix cache can never hit).
Every spelling of an added (control) token that occurs in the source text is broken
with a single space before tokenizing, so prompts and goldens are ordinary text tokens
only. BOS is prepended to every prompt.

Usage:
  python3 make_prompts.py --tokenizer tokenizer.json --deepinfer DIR --recipe DIR \
      --prose post1.md post2.md ... --out prompts.json
"""
import argparse
import json
import re
from pathlib import Path

from tokenizers import Tokenizer

LENS = [256, 1024, 4096, 16384]
GOLDEN = 256
BOS = 0


def read_tree(root: Path, suffix: str, subdirs: list[str]) -> str:
    parts = []
    for sub in subdirs:
        for f in sorted((root / sub).rglob(f"*{suffix}")):
            if "/vendor/" in str(f) or "/target/" in str(f):
                continue
            parts.append(f"# file: {f.relative_to(root)}\n{f.read_text(errors='replace')}\n")
    return "".join(parts)


def strip_post(md: str) -> str:
    md = re.sub(r"^---\n.*?\n---\n", "", md, count=1, flags=re.S)          # front matter
    md = re.sub(r"<style>.*?</style>", "", md, flags=re.S)                 # inline css
    md = re.sub(r"<script>.*?</script>", "", md, flags=re.S)
    md = re.sub(r"<[^>]+>", " ", md)                                        # html tags
    md = re.sub(r"\{%.*?%\}", "", md)                                       # liquid
    md = re.sub(r"[ \t]+", " ", md)
    return re.sub(r"\n{3,}", "\n\n", md)


def neutralise(text: str, spellings: list[str]) -> str:
    # Break every control-token spelling (e.g. "</think>" -> "< /think>") so the
    # tokenizer can only produce ordinary text tokens from it.
    for sp in sorted(spellings, key=len, reverse=True):
        if sp in text:
            text = text.replace(sp, sp[0] + " " + sp[1:])
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--deepinfer", required=True)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--prose", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tok = Tokenizer.from_file(a.tokenizer)
    added = tok.get_added_tokens_decoder()
    special = {int(i) for i in added}
    spellings = [v.content for v in added.values()]
    sources = {
        "code_py": read_tree(Path(a.deepinfer), ".py", ["deepinfer", "examples", "eval"]),
        "code_rs": read_tree(Path(a.recipe), ".rs", ["deepseek-recipe", "deepseek-recipe-core", "deepseek-recipe-encoding",
                                                     "deepseek-recipe-image", "deepseek-recipe-python", "server-rs"]),
        "prose": "\n\n".join(strip_post(Path(p).read_text()) for p in a.prose),
    }
    need = sum(LENS) + GOLDEN * len(LENS)
    prompts, golden, meta = [], [], []
    for name, text in sources.items():
        ids = tok.encode(neutralise(text, spellings), add_special_tokens=False).ids
        bad = [i for i in ids if i in special]
        assert not bad, f"{name}: {len(bad)} control tokens survived neutralisation"
        assert len(ids) >= need, f"{name}: only {len(ids)} tokens, need {need}"
        off = 0
        for L in LENS:
            p = [BOS] + ids[off:off + L - 1]
            g = ids[off + L - 1: off + L - 1 + GOLDEN]
            prompts.append(p); golden.append(g)
            meta.append({"source": name, "len": L, "offset": off})
            off += L - 1 + GOLDEN
        print(f"{name:8s} {len(ids):9,d} tokens available, {off:,d} used")
    Path(a.out).write_text(json.dumps({"lens": LENS, "golden_len": GOLDEN, "meta": meta, "prompts": prompts, "golden": golden}))
    print(f"wrote {a.out}: {len(prompts)} requests")


if __name__ == "__main__":
    main()
