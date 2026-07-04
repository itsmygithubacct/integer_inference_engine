"""trinote-import-bonsai-gguf — import PrismML Bonsai-8B Q1_0 GGUF into a Bonsai int-ref artifact."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..infer_int.artifact_io_bonsai import save_artifact_bonsai
from ..infer_int.import_bonsai_gguf import bonsai_gguf_provenance, import_bonsai_gguf_to_artifact
from ..notary_paths import default_gguf, default_artifact


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="trinote-import-bonsai-gguf",
                                 description="Import PrismML Bonsai-8B Q1_0 GGUF -> Bonsai int-ref artifact")
    ap.add_argument("--gguf", default=default_gguf())
    ap.add_argument("--out", default=default_artifact())
    ap.add_argument("--context-len", type=int, default=None,
                    help="RoPE table rows to commit (default: GGUF context_length)")
    ap.add_argument("--frac", type=int, default=16)
    ap.add_argument("--source", default="prism-ml/Bonsai-8B-gguf")
    ap.add_argument("--license", default="Apache-2.0")
    args = ap.parse_args(argv)

    art = import_bonsai_gguf_to_artifact(args.gguf, context_len=args.context_len, frac=args.frac)
    prov = bonsai_gguf_provenance(args.gguf, source=args.source, license=args.license)
    out = Path(args.out)
    digest = save_artifact_bonsai(art, out, provenance=prov)
    print(f"[bonsai-import] wrote {out}  sha256={digest}")
    print(f"[bonsai-import] source={prov['source']}  ggufSha256={prov['ggufSha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
