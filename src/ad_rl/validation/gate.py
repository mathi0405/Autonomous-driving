"""Predeclared paired regression rules and descriptive uncertainty intervals."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np


@dataclass(frozen=True)
class GateRules:
    """A versioned acceptance contract; tune on development data, then freeze."""

    max_new_collisions: int = 0
    max_new_offroads: int = 0
    max_new_failures: int = 0
    max_route_drop: float = 0.02
    max_lateral_increase_m: float = 0.10
    max_overspeed_increase_kmh: float = 3.0
    min_nominal_success_rate: float = 0.95

    def __post_init__(self) -> None:
        """Reject non-finite thresholds and fractional episode counts."""
        for key, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid gate threshold: {key}.")
        if self.min_nominal_success_rate > 1 or self.max_route_drop > 1:
            raise ValueError("Rate thresholds must be fractions.")
        for key in ("max_new_collisions", "max_new_offroads", "max_new_failures"):
            if not isinstance(getattr(self, key), int):
                raise ValueError(f"{key} must be an integer count.")


def load_rules(data: dict) -> GateRules:
    """Reject misspelled thresholds instead of silently disabling checks."""
    if set(data) - {f.name for f in fields(GateRules)}:
        raise ValueError("Unknown acceptance rule.")
    return GateRules(**data)


def compare(baseline: dict, candidate: dict, rules: GateRules | None = None) -> dict:
    """Compare paired cases only when simulator, suite and runtime are compatible."""
    rules = rules or GateRules()
    reasons = []
    for key in ("backend", "suite_sha256"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"Incomparable evidence: {key} differs.")
    if baseline["provenance"]["source_sha256"] != candidate["provenance"]["source_sha256"]:
        raise ValueError("Source differs: re-evaluate both policies with the same simulator.")
    for key in ("python", "platform"):
        if baseline["provenance"][key] != candidate["provenance"][key]:
            raise ValueError(f"Runtime differs: {key}.")
    for name in ("numpy", "gymnasium"):
        if (
            baseline["provenance"]["dependencies"][name]
            != candidate["provenance"]["dependencies"][name]
        ):
            raise ValueError(f"Runtime dependency differs: {name}.")
    a = {e["id"]: e["metrics"] for e in baseline["episodes"]}
    b = {e["id"]: e["metrics"] for e in candidate["episodes"]}
    if not a or set(a) != set(b):
        raise ValueError("Paired scenario IDs do not match.")
    cases = []
    for case_id in sorted(a):
        before, after = a[case_id], b[case_id]
        codes = []
        if after["collision"] and not before["collision"]:
            codes.append("NEW_COLLISION")
        if after["offroad"] and not before["offroad"]:
            codes.append("NEW_OFFROAD")
        if before["success"] and not after["success"]:
            codes.append("NEW_FAILURE")
        cases.append(
            {
                "id": case_id,
                "reason_codes": codes,
                "baseline_reason": before["terminal_reason"],
                "candidate_reason": after["terminal_reason"],
                "route_delta": after["route_completion"] - before["route_completion"],
                "lateral_delta_m": (
                    after["mean_abs_lateral_error_m"] - before["mean_abs_lateral_error_m"]
                ),
            }
        )
    for code, threshold in (
        ("NEW_COLLISION", rules.max_new_collisions),
        ("NEW_OFFROAD", rules.max_new_offroads),
        ("NEW_FAILURE", rules.max_new_failures),
    ):
        count = sum(code in case["reason_codes"] for case in cases)
        if count > threshold:
            reasons.append({"code": code, "observed": count, "limit": threshold})
    deltas = np.array([case["route_delta"] for case in cases])
    route_delta = float(np.mean(deltas))
    lateral_delta = float(np.mean([case["lateral_delta_m"] for case in cases]))
    if route_delta < -rules.max_route_drop:
        reasons.append(
            {
                "code": "ROUTE_COMPLETION_DROP",
                "observed": route_delta,
                "limit": -rules.max_route_drop,
            }
        )
    if lateral_delta > rules.max_lateral_increase_m:
        reasons.append(
            {
                "code": "LATERAL_ERROR_INCREASE",
                "observed": lateral_delta,
                "limit": rules.max_lateral_increase_m,
            }
        )
    targets = {s["id"]: s["target_speed_kmh"] for s in candidate["suite"]["scenarios"]}
    speed_delta = float(
        np.mean(
            [
                max(0.0, b[i]["mean_speed_kmh"] - targets[i])
                - max(0.0, a[i]["mean_speed_kmh"] - targets[i])
                for i in sorted(a)
            ]
        )
    )
    if speed_delta > rules.max_overspeed_increase_kmh:
        reasons.append(
            {
                "code": "OVERSPEED_REGRESSION",
                "observed": speed_delta,
                "limit": rules.max_overspeed_increase_kmh,
            }
        )
    nominal = candidate["by_regime"].get("nominal")
    if nominal and nominal["success_rate"] < rules.min_nominal_success_rate:
        reasons.append(
            {
                "code": "NOMINAL_SUCCESS_FLOOR",
                "observed": nominal["success_rate"],
                "limit": rules.min_nominal_success_rate,
            }
        )
    rng = np.random.default_rng(20261003)
    regimes = {s["id"]: s["regime"] for s in candidate["suite"]["scenarios"]}
    samples = np.zeros(2000)
    for regime in sorted(set(regimes.values())):
        group = np.array([case["route_delta"] for case in cases if regimes[case["id"]] == regime])
        samples += rng.choice(group, size=(2000, len(group)), replace=True).sum(axis=1) / len(cases)
    return {
        "schema": 1,
        "verdict": "FAIL" if reasons else "PASS",
        "reason_codes": reasons,
        "rules": asdict(rules),
        "paired_episodes": len(cases),
        "baseline_policy": baseline["policy"],
        "candidate_policy": candidate["policy"],
        "suite_sha256": candidate["suite_sha256"],
        "cases": cases,
        "route_delta": route_delta,
        "lateral_delta_m": lateral_delta,
        "overspeed_delta_kmh": speed_delta,
        "route_delta_paired_bootstrap_95ci": np.quantile(samples, [0.025, 0.975]).tolist(),
        "uncertainty_method": "Paired stratified bootstrap, 2000 resamples, fixed regime weights",
        "scope": "Finite paired kinematic simulation suite; not hardware validation.",
    }
