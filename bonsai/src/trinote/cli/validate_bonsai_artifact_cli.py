"""Validate a Bonsai artifact without materializing its weight tensors."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..infer_int.artifact_io import ArtifactError
from ..infer_int.artifact_io_bonsai import read_artifact_info_bonsai
from ..infer_int.bonsai_runtime import validate_bonsai35_receipt_identity


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="trinote-validate-bonsai-artifact",
        description=(
            "Validate safetensors structure, hash the complete artifact, and optionally "
            "bind it to the release identity and quality gate"
        ),
    )
    ap.add_argument("--artifact", required=True, help="Bonsai safetensors artifact")
    ap.add_argument("--architecture", default=None, help="required config.architecture")
    ap.add_argument(
        "--identity",
        default=None,
        help="Bonsai-27B release identity whose modelHash and quality gate must match",
    )
    args = ap.parse_args(argv)

    path = Path(args.artifact)
    try:
        info = read_artifact_info_bonsai(path)
        config = info.get("config")
        if not isinstance(config, dict):
            raise ValueError("artifact metadata config must be a JSON object")
        architecture = config.get("architecture")
        if not isinstance(architecture, str) or not architecture:
            raise ValueError("artifact metadata has no usable config.architecture")
        if args.architecture is not None and architecture != args.architecture:
            raise ValueError(
                f"artifact architecture {architecture!r} does not match required {args.architecture!r}"
            )
        digest = info["digest"]
        if args.identity is not None:
            if architecture != "qwen35":
                raise ValueError("release identity validation currently requires a qwen35 artifact")
            validate_bonsai35_receipt_identity(args.identity, digest)
    except (ArtifactError, OSError, ValueError, KeyError) as exc:
        print(f"[artifact] validation failed for {path}: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({
        "ok": True,
        "artifact": str(path),
        "architecture": architecture,
        "format": info["format"],
        "sha256": digest,
        "identity": args.identity,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
