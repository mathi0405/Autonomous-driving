"""Test measured-data interventions and regression cases found by frozen suites."""

import json
from pathlib import Path

import numpy as np
import pytest

from ad_rl.validation.policies import RobustController, policy_identity
from ad_rl.validation.runner import run_episode
from ad_rl.validation.safety import filter_action, predict_clearance
from ad_rl.validation.scenarios import Scenario


def test_centered_obstacle_triggers_braking_when_no_lane_safe_prediction_exists():
    action, decision = filter_action(
        np.array([0.0, 1.0]), 8, [{"forward_m": 8, "left_m": 0}], 0, 0, 0, 2
    )
    assert decision["intervened"] and action[1] <= 0
    assert decision["raw_action"] == [0, 1]


def test_clear_path_does_not_change_requested_action():
    requested = np.array([0.0, 0.2], dtype=np.float32)
    action, decision = filter_action(requested, 4, [], 0, 0, 0, 2)
    np.testing.assert_array_equal(action, requested)
    assert not decision["intervened"]


def test_future_collision_is_predicted_before_actual_contact():
    clearance = predict_clearance(0, 5, [{"forward_m": 6, "left_m": 0}])
    assert clearance < 0


def test_wide_road_current_sensor_branch_still_checks_obstacles():
    policy = RobustController(Scenario("wide", 1))
    info = {
        "lateral_error_m": 0.0,
        "heading_error_rad": 0.0,
        "speed_ms": 8.0,
        "curvature_inv_m": 0.0,
        "measurement_step": 0,
        "obstacle_detections": [{"forward_m": 8.0, "left_m": 0.0, "road_lateral_m": 0.0}],
    }
    action = policy(np.zeros(10), info)
    assert policy.last_decision["intervened"]
    assert action[1] <= 0


def test_command_history_is_bounded_without_losing_absolute_sensor_age():
    policy = RobustController(Scenario("long", 1))
    for step in range(150):
        info = {
            "lateral_error_m": 0.0,
            "heading_error_rad": 0.0,
            "speed_ms": 0.0,
            "curvature_inv_m": 0.0,
            "measurement_step": step,
            "obstacle_detections": [],
        }
        policy(np.zeros(10), info)
    assert len(policy.commands) == 101 and policy.steps == 150
    assert not policy.last_decision["intervened"]


@pytest.mark.parametrize(
    "case_id",
    ["sensing-002", "sensing-007", "sensing-010", "sensing-012", "sensing-033", "sensing-057"],
)
def test_newly_discovered_sensing_collisions_do_not_recur(case_id):
    root = Path(__file__).resolve().parents[1]
    saved = json.loads((root / "results/validation/third-held-out" / f"{case_id}.json").read_text())
    case = Scenario(**saved["scenario"])
    fixed = run_episode(case, policy_identity("robust"))
    assert not fixed["metrics"]["collision"]
    assert fixed["metrics"]["safety_interventions"] > 0
