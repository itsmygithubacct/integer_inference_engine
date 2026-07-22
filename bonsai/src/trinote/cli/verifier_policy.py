"""Deterministic routing for exact receipt verification.

Policies are intentionally small JSON documents produced by the verifier
benchmark.  A route selects an engine implementation and an exact full replay
algorithm; both choices preserve the receipt verifier's trust semantics.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


SCHEMA = "receipt-verifier-policy/v1"
ENGINES = {"oracle", "native"}
STRATEGIES = {"teacher-forced", "cached-replay"}


def load_verifier_policy(path: str | Path) -> dict[str, Any]:
    policy = json.loads(Path(path).read_text("utf-8"))
    validate_verifier_policy(policy)
    return policy


def validate_verifier_policy(policy: dict[str, Any]) -> None:
    if not isinstance(policy, dict) or policy.get("schema") != SCHEMA:
        raise ValueError(f"verifier policy must use schema {SCHEMA!r}")
    artifact_digest = policy.get("artifactSha256")
    if not isinstance(artifact_digest, str) or re.fullmatch(r"[0-9a-f]{64}", artifact_digest) is None:
        raise ValueError("verifier policy artifactSha256 must be a lowercase SHA-256 digest")
    evidence_digest = policy.get("evidenceSha256")
    if evidence_digest is not None and (
        not isinstance(evidence_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", evidence_digest) is None
    ):
        raise ValueError("verifier policy evidenceSha256 must be a lowercase SHA-256 digest")
    threads = policy.get("threads")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
        raise ValueError("verifier policy threads must be a positive integer")
    rules = policy.get("rules")
    if not isinstance(rules, list):
        raise ValueError("verifier policy rules must be a list")
    for index, rule in enumerate(rules):
        _validate_route(rule, f"rules[{index}]")
        for key in ("minInputTokens", "maxInputTokens", "minOutputTokens", "maxOutputTokens"):
            if key in rule and (
                isinstance(rule[key], bool) or not isinstance(rule[key], int) or int(rule[key]) < 0
            ):
                raise ValueError(f"verifier policy {key} in rules[{index}] must be a non-negative integer")
        for lower, upper in (
            ("minInputTokens", "maxInputTokens"),
            ("minOutputTokens", "maxOutputTokens"),
        ):
            if lower in rule and upper in rule and rule[lower] > rule[upper]:
                raise ValueError(
                    f"verifier policy {lower} exceeds {upper} in rules[{index}]"
                )
    default = policy.get("default")
    if not isinstance(default, dict):
        raise ValueError("verifier policy requires a default route")
    _validate_route(default, "default")
    measured_points = policy.get("measuredPoints")
    if measured_points is not None:
        if not isinstance(measured_points, list) or not measured_points:
            raise ValueError("verifier policy measuredPoints must be a non-empty list")
        for index, point in enumerate(measured_points):
            if not isinstance(point, dict):
                raise ValueError(f"verifier policy measuredPoints[{index}] must be an object")
            for key in ("inputTokens", "outputTokens"):
                value = point.get(key)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(
                        f"verifier policy measuredPoints[{index}].{key} must be a non-negative integer"
                    )
    require_measured = policy.get("requireMeasuredPoint", False)
    if not isinstance(require_measured, bool):
        raise ValueError("verifier policy requireMeasuredPoint must be boolean")
    if require_measured and not measured_points:
        raise ValueError("requireMeasuredPoint needs measuredPoints evidence")
    if require_measured:
        measured: set[tuple[int, int]] = set()
        for index, point in enumerate(measured_points):
            key = (int(point["inputTokens"]), int(point["outputTokens"]))
            if key in measured:
                raise ValueError(f"verifier policy measuredPoints[{index}] is duplicated")
            measured.add(key)
            matching = [rule for rule in rules if _matches(rule, *key)]
            if not matching or not _is_exact_point_rule(matching[0], *key):
                raise ValueError(
                    "every measured point must select a first-match rule with exact input/output bounds"
                )
            measured_threads = matching[0].get("measuredThreads")
            if (
                isinstance(measured_threads, bool)
                or not isinstance(measured_threads, int)
                or measured_threads != threads
            ):
                raise ValueError(
                    "every measured-point rule must attest measuredThreads equal to policy threads"
                )


def route_verification(policy: dict[str, Any], *, input_tokens: int, output_tokens: int) -> dict[str, str]:
    """Return the first matching route; rule order is load-bearing and deterministic."""
    validate_verifier_policy(policy)
    n_input = int(input_tokens)
    n_output = int(output_tokens)
    if n_input < 0 or n_output < 0:
        raise ValueError("committed token counts must be non-negative")
    if policy.get("requireMeasuredPoint") is True:
        measured = {
            (int(point["inputTokens"]), int(point["outputTokens"]))
            for point in policy["measuredPoints"]
        }
        if (n_input, n_output) not in measured:
            raise ValueError(
                "committed token counts are outside the verifier policy's measured matrix: "
                f"input={n_input} output={n_output}"
            )
    for rule in policy["rules"]:
        if not _matches(rule, n_input, n_output):
            continue
        return {"engine": str(rule["engine"]), "strategy": str(rule["strategy"])}
    default = policy["default"]
    return {"engine": str(default["engine"]), "strategy": str(default["strategy"])}


def _validate_route(route: dict[str, Any], label: str) -> None:
    if not isinstance(route, dict):
        raise ValueError(f"verifier policy {label} must be an object")
    if route.get("engine") not in ENGINES:
        raise ValueError(f"verifier policy {label} engine must be one of {sorted(ENGINES)}")
    if route.get("strategy") not in STRATEGIES:
        raise ValueError(f"verifier policy {label} strategy must be one of {sorted(STRATEGIES)}")


def _matches(rule: dict[str, Any], input_tokens: int, output_tokens: int) -> bool:
    bounds = (
        ("minInputTokens", input_tokens, lambda value, bound: value >= bound),
        ("maxInputTokens", input_tokens, lambda value, bound: value <= bound),
        ("minOutputTokens", output_tokens, lambda value, bound: value >= bound),
        ("maxOutputTokens", output_tokens, lambda value, bound: value <= bound),
    )
    return all(key not in rule or predicate(value, int(rule[key])) for key, value, predicate in bounds)


def _is_exact_point_rule(rule: dict[str, Any], input_tokens: int, output_tokens: int) -> bool:
    return (
        rule.get("minInputTokens") == input_tokens
        and rule.get("maxInputTokens") == input_tokens
        and rule.get("minOutputTokens") == output_tokens
        and rule.get("maxOutputTokens") == output_tokens
    )
