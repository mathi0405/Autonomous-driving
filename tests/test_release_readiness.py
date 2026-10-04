"""Absolute simulation release checks must not inherit baseline failures."""

import copy
from dataclasses import asdict

import pytest

from ad_rl.validation.cli import main
from ad_rl.validation.gate import compare
from ad_rl.validation.policies import policy_identity
from ad_rl.validation.readiness import ReleaseRequirements, assess_readiness
from ad_rl.validation.runner import run_suite
from ad_rl.validation.scenarios import Scenario


@pytest.fixture
def completed_run(tmp_path):
    cases = [
        Scenario(f"{r}-0", 1, regime=r, num_obstacles=0)
        for r in ("nominal", "geometry", "sensing", "actuator")
    ]
    suite = {"schema": 1, "name": "test", "scenarios": [asdict(c) for c in cases]}
    return run_suite(suite, cases, policy_identity("stanley"), tmp_path / "completed")


def test_complete_fault_free_evidence_can_meet_simulation_requirements(completed_run):
    verdict = assess_readiness(completed_run, ReleaseRequirements(min_cases_per_regime=1))
    assert verdict["status"] == "SIMULATION_REQUIREMENTS_MET"
    assert verdict["hardware_release_approved"] is False


def test_default_requirements_reject_tiny_demo(completed_run):
    verdict = assess_readiness(completed_run)
    assert verdict["status"] == "BLOCKED"
    assert {r["code"] for r in verdict["blockers"]} == {"INSUFFICIENT_REGIME_EVIDENCE"}


def test_relative_pass_with_shared_collision_still_blocks_release(tmp_path):
    case = Scenario("actuator-0", 1, regime="actuator", curviness=0, obstacle_offset_m=0)
    suite = {"schema": 1, "scenarios": [asdict(case)]}
    run = run_suite(suite, [case], policy_identity("stanley"), tmp_path / "collision")
    assert run["summary"]["collision_rate"] == 1
    assert compare(run, run)["verdict"] == "PASS"
    result = assess_readiness(run, ReleaseRequirements(min_cases_per_regime=1))
    assert result["status"] == "BLOCKED"
    assert "COLLISIONS_PRESENT" in {r["code"] for r in result["blockers"]}
    assert "MISSING_REQUIRED_REGIME" in {r["code"] for r in result["blockers"]}


def test_average_speed_cannot_hide_peak_overspeed(completed_run):
    run = copy.deepcopy(completed_run)
    run["episodes"][0]["metrics"]["peak_speed_kmh"] = 40.0
    result = assess_readiness(run, ReleaseRequirements(min_cases_per_regime=1))
    assert "PEAK_OVERSPEED" in {r["code"] for r in result["blockers"]}


def test_readiness_command_has_machine_readable_blocked_exit(tmp_path, completed_run):
    path = tmp_path / "completed/run.json"
    status = main(["readiness", "--run", str(path), "--out", str(tmp_path / "readiness")])
    assert status == 1 and (tmp_path / "readiness/readiness.json").exists()


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_release_sample_budget_is_rejected(size):
    with pytest.raises(ValueError):
        ReleaseRequirements(min_cases_per_regime=size)
