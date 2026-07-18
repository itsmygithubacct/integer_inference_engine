"""Generate or verify compact Bonsai-27B integer parity traces."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..infer_int.artifact_io_bonsai import load_artifact_bonsai
from ..infer_int.trace_bonsai35 import (
    assert_trace_equal,
    trace_cached_greedy,
    trace_prefill,
)
from ..notary_paths import notary_home


def _token_ids(raw: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--tokens must be comma-separated integers") from exc
    if not values or any(v < 0 for v in values):
        raise argparse.ArgumentTypeError("--tokens must contain at least one non-negative ID")
    return values


def main(argv: list[str] | None = None) -> int:
    home = Path(notary_home())
    ap = argparse.ArgumentParser(
        prog="trinote-trace-bonsai35",
        description="Generate/verify canonical layer and cache hashes for Bonsai-27B",
    )
    ap.add_argument(
        "--artifact",
        default=str(home / "models" / "Bonsai-27B-Q1_0-int-qwen35.safetensors"),
    )
    ap.add_argument("--tokens", type=_token_ids, required=True)
    ap.add_argument("--decode", type=int, default=0,
                    help="also trace this many cached greedy output tokens")
    ap.add_argument("--native", action="store_true",
                    help="use native producers (the default is the canonical NumPy oracle)")
    ap.add_argument("--full-layers", action="store_true",
                    help="with --decode, retain every layer row instead of aggregate commitments")
    ap.add_argument("--out", default="-", help="JSON output path, or - for stdout")
    ap.add_argument("--verify", default=None, help="compare with an existing trace JSON")
    args = ap.parse_args(argv)
    if args.decode < 0:
        ap.error("--decode must be >= 0")

    artifact, info = load_artifact_bonsai(args.artifact)
    if str(artifact.get("config", {}).get("architecture")) != "qwen35":
        raise ValueError("trace CLI requires a Qwen3.5 Bonsai artifact")
    if args.decode:
        trace = trace_cached_greedy(
            artifact,
            args.tokens,
            args.decode,
            native=args.native,
            include_layer_traces=args.full_layers,
        )
    else:
        trace = trace_prefill(artifact, args.tokens, native=args.native)
    trace["artifactSha256"] = info["digest"]
    rendered = json.dumps(trace, sort_keys=True, indent=2) + "\n"

    if args.verify:
        expected = json.loads(Path(args.verify).read_text())
        assert_trace_equal(trace, expected)
        print(f"[bonsai35-trace] VERIFIED {args.verify}", file=sys.stderr)
    if args.out == "-":
        sys.stdout.write(rendered)
    else:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered)
        print(f"[bonsai35-trace] wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
