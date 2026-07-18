"""CLI for importing the Bonsai-27B Qwen3.5 GGUF into a native artifact."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ..infer_int.artifact_io_bonsai import save_artifact_bonsai
from ..infer_int.import_bonsai35_gguf import (
    bonsai35_gguf_provenance,
    import_bonsai35_gguf_to_artifact,
)
from ..notary_paths import notary_home


def main(argv: list[str] | None = None) -> int:
    home = Path(notary_home())
    ap = argparse.ArgumentParser(
        prog="trinote-import-bonsai35-gguf",
        description="Import prism-ml/Bonsai-27B Q1_0 GGUF into the deterministic integer engine",
    )
    ap.add_argument(
        "--gguf",
        default=str(home / "models" / "Bonsai-27B-Q1_0.gguf"),
        help="source prism-ml/Bonsai-27B Q1_0 GGUF",
    )
    ap.add_argument(
        "--out",
        default=str(home / "models" / "Bonsai-27B-Q1_0-int-qwen35.safetensors"),
        help="destination native safetensors artifact",
    )
    ap.add_argument("--context-len", type=int, default=4096)
    ap.add_argument("--frac", type=int, default=16)
    args = ap.parse_args(argv)

    t0 = time.time()
    print(f"[bonsai35-import] reading {args.gguf}", file=sys.stderr)
    artifact = import_bonsai35_gguf_to_artifact(
        args.gguf,
        context_len=args.context_len,
        frac=args.frac,
        progress=lambda msg: print(msg, file=sys.stderr, flush=True),
    )
    print(f"[bonsai35-import] writing {args.out}", file=sys.stderr, flush=True)
    digest = save_artifact_bonsai(
        artifact,
        args.out,
        provenance=bonsai35_gguf_provenance(args.gguf),
    )
    size = Path(args.out).stat().st_size
    print(
        f"[bonsai35-import] wrote {size} bytes sha256={digest} in {time.time() - t0:.1f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
