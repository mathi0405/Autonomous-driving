"""Absolute simulation release requirements, separate from relative regression gates."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from ad_rl.validation.runner import validate_outcomes

REQUIRED_REGIMES = ("nominal", "geometry", "sensing", "actuator")


@dataclass(frozen=True)
class ReleaseRequirements:
    """Predeclared finite-suite requirements; these cannot approve hardware deployment."""

    min_cases_per_regime: int = 32
    max_peak_overspeed_kmh: float = 3.0

    def __post_init__(self) -> None:
        """Require a positive sample count and a finite nonnegative speed tolerance."""
        if type(self.min_cases_per_regime) is not int or self.min_cases_per_regime < 1:
            raise ValueError("Release sample count must be a positive integer.")
        if (
            isinstance(self.max_peak_overspeed_kmh, bool)
            or not isinstance(self.max_peak_overspeed_kmh, int | float)
            or not math.isfinite(self.max_peak_overspeed_kmh)
            or self.max_peak_overspeed_kmh < 0
        ):
            raise ValueError("Release speed tolerance must be finite and nonnegative.")


def assess_readiness(run: dict, requirements: ReleaseRequirements | None = None) -> dict:
    """Report all absolute blockers from verified evidence without consulting a baseline."""
    requirements = requirements or ReleaseRequirements()
    validate_outcomes(run["episodes"])
    scenarios = {s["id"]: s for s in run["suite"]["scenarios"]}
    episodes = {e["id"]: e for e in run["episodes"]}
    if (
        len(scenarios) != len(run["suite"]["scenarios"])
        or len(episodes) != len(run["episodes"])
        or set(episodes) != set(scenarios)
    ):
        raise ValueError("Release scenario mapping is incomplete or duplicated.")
    blockers = []
    counts = {
        regime: sum(s["regime"] == regime for s in scenarios.values())
        for regime in REQUIRED_REGIMES
    }
    for regime, count in counts.items():
        if count < requirements.min_cases_per_regime:
            blockers.append(
                {
                    "code": (
                        "MISSING_REQUIRED_REGIME" if count == 0 else "INSUFFICIENT_REGIME_EVIDENCE"
                    ),
                    "regime": regime,
                    "observed": count,
                    "required": requirements.min_cases_per_regime,
                }
            )
    failed_ids = []
    for flag, code in (
        ("collision", "COLLISIONS_PRESENT"),
        ("offroad", "LANE_DEPARTURES_PRESENT"),
        ("timeout", "TIMEOUTS_PRESENT"),
    ):
        ids = sorted(i for i, e in episodes.items() if e["metrics"][flag])
        failed_ids.extend(ids)
        if ids:
            blockers.append({"code": code, "observed": len(ids), "limit": 0, "scenario_ids": ids})
    speeding, missing_speed = [], []
    for case_id, episode in episodes.items():
        peak = episode["metrics"].get("peak_speed_kmh")
        if peak is None:
            missing_speed.append(case_id)
        elif (
            isinstance(peak, bool)
            or not isinstance(peak, int | float)
            or not math.isfinite(peak)
            or peak < 0
        ):
            raise ValueError("Invalid peak speed evidence.")
        elif peak > scenarios[case_id]["target_speed_kmh"] + requirements.max_peak_overspeed_kmh:
            speeding.append(case_id)
    for code, ids in (("MISSING_SPEED_EVIDENCE", missing_speed), ("PEAK_OVERSPEED", speeding)):
        if ids:
            blockers.append(
                {"code": code, "observed": len(ids), "limit": 0, "scenario_ids": sorted(ids)}
            )
    return {
        "schema": 1,
        "status": "BLOCKED" if blockers else "SIMULATION_REQUIREMENTS_MET",
        "hardware_release_approved": False,
        "requirements": {
            **asdict(requirements),
            "required_regimes": list(REQUIRED_REGIMES),
            "max_failed_missions": 0,
        },
        "blockers": blockers,
        "failed_scenario_ids": sorted(failed_ids),
        "cases_per_regime": counts,
        "suite_sha256": run["suite_sha256"],
        "source_sha256": run["provenance"]["source_sha256"],
        "scope": "Finite kinematic simulation only; CARLA and hardware remain unvalidated.",
    }
