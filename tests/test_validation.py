"""Behavioral and evidence-integrity checks for the regression platform."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict

import numpy as np
import pytest

from ad_rl.envs.fallback_env import KinematicDrivingEnv
from ad_rl.utils.config import EnvConfig
from ad_rl.validation.cli import main, search_failures
from ad_rl.validation.environment import ScenarioEnvironment
from ad_rl.validation.gate import compare
from ad_rl.validation.policies import policy_identity
from ad_rl.validation.runner import load_run, run_episode, run_suite, write_json
from ad_rl.validation.scenarios import Scenario, generate_suite, load_suite


def test_collision_near_finish_is_not_success():
    env = KinematicDrivingEnv(EnvConfig(observation="state", frame_stack=1, action_repeat=1))
    env.reset(seed=7)
    env._idx = len(env._path_xy) - 2
    env._x, env._y = env._path_xy[env._idx]
    env._yaw = env._path_yaw[env._idx]
    env._obstacles = np.array([[env._x, env._y]])
    _, reward, terminated, _, info = env.step(np.array([0, 0]))
    assert terminated and info["collision"] and not info["is_success"]
    assert info["terminal_reason"] == "COLLISION"
    assert "goal" not in info["reward_components"]
    assert reward < 0


def test_disturbed_sensor_does_not_change_scoring_state():
    scenario = Scenario("biased", 12, lateral_bias_m=0.5, sensor_noise_std_m=0)
    env = ScenarioEnvironment(scenario)
    obs, info = env.reset()
    measured, measured_info = env.measured(obs, info)
    assert measured_info["lateral_error_m"] == pytest.approx(0.5)
    assert measured[1] == pytest.approx(0.25)
    assert info["lateral_error_m"] == 0 and obs[1] == 0
    assert measured_info is not info


def test_observation_delay_is_real():
    env = ScenarioEnvironment(Scenario("delay", 12, observation_delay_steps=2))
    obs, info = env.reset()
    first = env.measured(obs, info)
    obs, _, _, _, info = env.step(np.array([0.4, 0]))
    delayed = env.measured(obs, info)
    assert delayed[1]["measurement_step"] == first[1]["measurement_step"]
    assert info["measurement_step"] != delayed[1]["measurement_step"]


def test_episode_repeats_exactly_with_noise_and_delay():
    scenario = Scenario("repeat", 444, sensor_noise_std_m=0.04, observation_delay_steps=2)
    a = run_episode(scenario, policy_identity("stanley"))
    b = run_episode(scenario, policy_identity("stanley"))
    assert a["content_sha256"] == b["content_sha256"]
    assert a["trajectory"] == b["trajectory"]


@pytest.fixture
def nominal_suite():
    cases = [Scenario(f"nominal-{i}", seed) for i, seed in enumerate([1, 7, 23, 20240])]
    suite = {"schema": 1, "name": "test", "scenarios": [asdict(s) for s in cases]}
    return suite, cases


def test_identical_controller_passes_gate(tmp_path, nominal_suite):
    suite, cases = nominal_suite
    a = run_suite(suite, cases, policy_identity("stanley"), tmp_path / "baseline")
    b = run_suite(suite, cases, policy_identity("stanley"), tmp_path / "candidate")
    result = compare(a, b)
    assert result["verdict"] == "PASS"
    assert result["route_delta_paired_bootstrap_95ci"] == [0, 0]


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("mutant-steering", "NEW_OFFROAD"),
        ("mutant-speed", "OVERSPEED_REGRESSION"),
        ("mutant-stop", "NEW_FAILURE"),
    ],
)
def test_gate_catches_behavioral_mutations(tmp_path, nominal_suite, mutation, expected):
    suite, cases = nominal_suite
    a = run_suite(suite, cases, policy_identity("stanley"), tmp_path / "baseline")
    b = run_suite(suite, cases, policy_identity(mutation), tmp_path / "candidate")
    result = compare(a, b)
    assert result["verdict"] == "FAIL"
    assert expected in {r["code"] for r in result["reason_codes"]}


def test_tampered_episode_is_rejected(tmp_path, nominal_suite):
    suite, cases = nominal_suite
    run_suite(suite, cases, policy_identity("pid"), tmp_path / "run")
    path = tmp_path / "run" / "episodes" / "nominal-0.json"
    episode = json.loads(path.read_text())
    episode["metrics"]["success"] = False
    write_json(path, episode)
    with pytest.raises(ValueError, match="integrity"):
        load_run(tmp_path / "run" / "run.json")


def test_tampered_summary_is_rejected(tmp_path, nominal_suite):
    suite, cases = nominal_suite
    result = run_suite(suite, cases, policy_identity("pid"), tmp_path / "run")
    result["summary"]["success_rate"] = 0.0
    write_json(tmp_path / "run" / "run.json", result)
    with pytest.raises(ValueError, match="summary"):
        load_run(tmp_path / "run" / "run.json")


def test_invalid_pair_cannot_pass(tmp_path, nominal_suite):
    suite, cases = nominal_suite
    baseline = run_suite(suite, cases, policy_identity("pid"), tmp_path / "baseline")
    other = copy.deepcopy(baseline)
    other["suite_sha256"] = "wrong"
    with pytest.raises(ValueError, match="Incomparable"):
        compare(baseline, other)
    other = copy.deepcopy(baseline)
    other["provenance"]["source_sha256"] = "changed"
    with pytest.raises(ValueError, match="Source differs"):
        compare(baseline, other)


def test_bundle_cannot_be_overwritten(tmp_path, nominal_suite):
    suite, cases = nominal_suite
    run_suite(suite, cases, policy_identity("pid"), tmp_path / "run")
    with pytest.raises(FileExistsError):
        run_suite(suite, cases, policy_identity("pid"), tmp_path / "run")


@pytest.mark.parametrize(
    "settings",
    [
        {"max_steps": 0},
        {"sensor_noise_std_m": float("nan")},
        {"observation_delay_steps": 1.2},
        {"seed": -1},
    ],
)
def test_invalid_scenario_is_rejected(settings):
    defaults = {"id": "case", "seed": 3}
    defaults.update(settings)
    with pytest.raises(ValueError):
        Scenario(**defaults)


def test_suite_rejects_unknown_fields_and_duplicates(tmp_path):
    data = generate_suite(123, 1, "development")
    data["scenarios"][0]["misspelled_fault"] = 1
    path = tmp_path / "suite.json"
    write_json(path, data)
    with pytest.raises(ValueError, match="Unknown"):
        load_suite(path)
    del data["scenarios"][0]["misspelled_fault"]
    data["scenarios"].append(data["scenarios"][0])
    write_json(path, data)
    with pytest.raises(ValueError, match="Duplicate"):
        load_suite(path)


def test_cli_invalid_input_returns_two(tmp_path):
    status = main(
        [
            "run",
            "--suite",
            str(tmp_path / "absent"),
            "--policy",
            "pid",
            "--out",
            str(tmp_path / "run"),
        ]
    )
    assert status == 2


def test_bounded_search_finds_and_replays_failure(tmp_path):
    result = search_failures("stanley", 20261003, 24, tmp_path / "search")
    assert result["failure_found"] and result["episodes_used"] <= 24
    failure = json.loads((tmp_path / "search" / "failure.json").read_text())
    repeated = run_episode(Scenario(**failure["scenario"]), failure["policy"])
    assert repeated["content_sha256"] == result["failure_sha256"]
    assert not repeated["metrics"]["success"]
