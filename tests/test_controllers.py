"""Tests for the classical control baselines."""

from __future__ import annotations

from ad_rl.control import PID, VehiclePIDController
from ad_rl.envs import make_env
from helpers import effective_dt, make_cfg


def test_pid_basic_response_and_reset():
    pid = PID(kp=1.0, ki=0.0, kd=0.0, dt=0.1)
    assert pid.step(1.0) > 0
    assert pid.step(-1.0) < 0
    pid.reset()
    assert pid._integral == 0.0
    assert pid._prev_error == 0.0


def test_pid_output_is_clipped():
    pid = PID(kp=100.0, ki=0.0, kd=0.0, dt=0.1, output_clip=(-1.0, 1.0))
    assert pid.step(10.0) == 1.0
    assert pid.step(-10.0) == -1.0


def test_pid_controller_solves_fallback():
    cfg = make_cfg("state", max_steps=600)
    successes = 0
    for ep in range(5):
        env = make_env("fallback", cfg)
        _, info = env.reset(seed=300 + ep)
        ctrl = VehiclePIDController(cfg.reward.target_speed_ms, dt=effective_dt(cfg))
        ctrl.reset()
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(ctrl.act_from_info(info))
            done = terminated or truncated
        successes += int(info.get("is_success", False))
    assert successes >= 4


def test_stanley_makes_progress():
    cfg = make_cfg("state", max_steps=600)
    env = make_env("fallback", cfg)
    _, info = env.reset(seed=7)
    ctrl = VehiclePIDController(cfg.reward.target_speed_ms, dt=effective_dt(cfg), lateral="stanley")
    ctrl.reset()
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step(ctrl.act_from_info(info))
        done = terminated or truncated
    assert info.get("route_fraction", 0.0) > 0.5
