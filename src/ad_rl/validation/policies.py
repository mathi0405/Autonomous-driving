"""Controller identities, explicit mutation controls, and optional saved RL policies."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ad_rl.control.controllers import VehiclePIDController
from ad_rl.validation.scenarios import Scenario

CONTROLLERS = (
    "pid",
    "stanley",
    "preview",
    "robust",
    "mutant-steering",
    "mutant-speed",
    "mutant-stop",
)


class RobustController:
    """Latency prediction, curvature speed planning and local obstacle avoidance.

    Consumes synthetic state measurements and limited-range obstacle detections.
    No scenario seed, true vehicle pose, ground-truth clearance or fault magnitude
    is available to the control law. This is not a camera-based controller.
    """

    def __init__(self, scenario: Scenario) -> None:
        self.target_speed_ms = scenario.target_speed_kmh / 3.6
        self.road_half_width_m = scenario.road_half_width_m
        self.speed = VehiclePIDController(self.target_speed_ms, dt=0.2)
        self.commands: list[np.ndarray] = []
        self.integral = 0.0
        self.offset = 0.0

    def __call__(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """Predict delayed state and follow a locally displaced lane reference."""
        lateral = float(info["lateral_error_m"])
        heading = float(info["heading_error_rad"])
        speed = float(info["speed_ms"])
        curvature = float(info["curvature_inv_m"])
        age = max(0, len(self.commands) - int(info["measurement_step"]))
        for command in self.commands[-age:] if age else []:
            for _ in range(2):
                lateral += speed * np.sin(heading) * 0.1
                heading += (speed / 2.8 * np.tan(float(command[0]) * 0.5) - speed * curvature) * 0.1
                acceleration = float(command[1]) * (3 if command[1] >= 0 else 6)
                speed = max(0, speed + acceleration * 0.1)
        desired = 0.0
        obstacles = sorted(info["obstacle_detections"], key=lambda o: o["forward_m"])
        for obstacle in obstacles:
            forward = obstacle["forward_m"] - speed * age * 0.2
            road_left = obstacle["road_lateral_m"]
            if -1.5 < forward < 24 and abs(road_left) < 1.4:
                desired = (-1 if road_left >= 0 else 1) * min(0.8, max(0.3, 1.65 - abs(road_left)))
                break
        self.offset += np.clip(desired - self.offset, -0.08, 0.08)
        error = lateral - self.offset
        self.integral = float(np.clip(self.integral + error * 0.2, -1.5, 1.5))
        steering_angle = (
            np.arctan(2.8 * curvature)
            - 1.0 * heading
            - np.arctan2(1.6 * error, max(2.0, speed))
            - 0.06 * self.integral
        )
        steer = float(np.clip(steering_angle / 0.5, -1, 1))
        upcoming = max(abs(curvature), float(np.max(np.abs(obs[5:]))) / 50.0)
        desired_speed = min(self.target_speed_ms, np.sqrt(1.8 / max(0.003, upcoming)))
        if age:
            desired_speed = min(desired_speed, 6.0)
        if abs(error) > 0.5:
            desired_speed = min(desired_speed, 4.0)
        if desired:
            desired_speed = min(desired_speed, 3.0)
        throttle = self.speed._lon.step(desired_speed - speed)
        action = np.array([steer, throttle], dtype=np.float32)
        self.commands.append(action.copy())
        return action


class ValidationPolicy:
    """A reset-per-episode controller with intentional regression variants."""

    def __init__(self, name: str, scenario: Scenario) -> None:
        if name not in CONTROLLERS:
            raise ValueError(f"Unknown controller {name}.")
        self.name = name
        self.controller = VehiclePIDController(
            scenario.target_speed_kmh / 3.6, dt=0.2, lateral="stanley"
        )
        if name == "pid":
            self.controller.lateral_mode = "pid"
        if name == "mutant-speed":
            self.controller.target_speed_ms = 50 / 3.6

    def __call__(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """Drive using measured state; preview adds a geometric feedforward term."""
        action = self.controller.act_from_info(info)
        if self.name == "preview":
            curvature = float(obs[5]) / 50.0
            feedforward = float(np.arctan(2.8 * curvature) / 0.5)
            action[0] = np.clip(action[0] + feedforward, -1, 1)
        elif self.name == "mutant-steering":
            action[0] *= -1
        elif self.name == "mutant-stop":
            action[:] = [0, -1]
        return action


def policy_identity(name: str, model: str | None = None, algo: str = "ppo") -> dict:
    """Identify built-in controllers or the exact bytes of a saved model."""
    if name == "model":
        if not model or not Path(model).is_file():
            raise ValueError("Model policy requires an existing --model file.")
        digest = hashlib.sha256(Path(model).read_bytes()).hexdigest()
        return {
            "name": name,
            "algorithm": algo,
            "model_sha256": digest,
            "model_path": str(Path(model).resolve()),
        }
    if name not in CONTROLLERS:
        raise ValueError(f"Unknown policy {name}.")
    return {"name": name}


def make_policy(identity: dict, scenario: Scenario):
    """Construct a fresh policy so controller state never leaks across scenarios."""
    if identity["name"] != "model":
        if identity["name"] == "robust":
            return RobustController(scenario)
        return ValidationPolicy(identity["name"], scenario)
    from ad_rl.agents import load_agent

    path = identity["model_path"]
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != identity["model_sha256"]:
        raise ValueError("Model bytes no longer match the recorded identity.")
    model = load_agent(identity["algorithm"], path)
    if model.observation_space.shape != (10,):
        raise ValueError("This validation suite requires an unstacked state policy (10 values).")

    def predict(obs, info):
        return model.predict(obs, deterministic=True)[0]

    return predict
