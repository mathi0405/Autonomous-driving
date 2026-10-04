"""Controller identities, explicit mutation controls, and optional saved RL policies."""

from __future__ import annotations

import hashlib
from collections import deque
from pathlib import Path

import numpy as np

from ad_rl.control.controllers import VehiclePIDController
from ad_rl.validation.safety import filter_action
from ad_rl.validation.scenarios import Scenario

CONTROLLERS = (
    "pid",
    "stanley",
    "preview",
    "robust",
    "robust-v1",
    "robust-v2",
    "robust-v3",
    "robust-v4",
    "robust-v5",
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

    def __init__(self, scenario: Scenario, version: int = 6) -> None:
        self.target_speed_ms = scenario.target_speed_kmh / 3.6
        self.road_half_width_m = scenario.road_half_width_m
        self.speed = VehiclePIDController(self.target_speed_ms, dt=0.2)
        self.commands: deque[np.ndarray] = deque(maxlen=101)
        self.steps = 0
        self.integral = 0.0
        self.offset = 0.0
        self.version = version
        self.nominal = VehiclePIDController(self.target_speed_ms, dt=0.2, lateral="stanley")
        self.last_decision: dict = {}

    def __call__(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """Predict delayed state and follow a locally displaced lane reference."""
        lateral = float(info["lateral_error_m"])
        heading = float(info["heading_error_rad"])
        speed = float(info["speed_ms"])
        curvature = float(info["curvature_inv_m"])
        if self.version >= 5:
            landmarks = [o for o in info["obstacle_detections"] if -1 < o["forward_m"] < 8]
            if landmarks:
                landmark = min(landmarks, key=lambda o: abs(o["forward_m"]))
                forward, left = landmark["forward_m"], landmark["left_m"]
                offset = (
                    landmark["road_lateral_m"]
                    - forward * np.sin(heading)
                    - left * np.cos(heading)
                    + 0.5 * curvature * forward**2
                )
                weight = max(0.0, min(1.0, (8 - abs(forward)) / 5))
                lateral = (1 - weight) * lateral + weight * offset
        age = max(0, self.steps - int(info["measurement_step"]))
        self.steps += 1
        if self.version >= 3 and age == 0 and self.road_half_width_m >= 1.8:
            action = self.nominal.act_from_info(info)
            if self.version >= 6:
                action, self.last_decision = filter_action(
                    action,
                    speed,
                    info["obstacle_detections"],
                    lateral,
                    heading,
                    curvature,
                    self.road_half_width_m,
                )
            self.commands.append(action.copy())
            return action
        shift_x, shift_y, shift_yaw = 0.0, 0.0, 0.0
        for command in list(self.commands)[-age:] if age else []:
            for _ in range(2):
                lateral += speed * np.sin(heading) * 0.1
                heading += (speed / 2.8 * np.tan(float(command[0]) * 0.5) - speed * curvature) * 0.1
                acceleration = float(command[1]) * (3 if command[1] >= 0 else 6)
                speed = max(0, speed + acceleration * 0.1)
                shift_x += speed * np.cos(shift_yaw) * 0.1
                shift_y += speed * np.sin(shift_yaw) * 0.1
                shift_yaw += speed / 2.8 * np.tan(float(command[0]) * 0.5) * 0.1
        desired = 0.0
        obstacles = sorted(info["obstacle_detections"], key=lambda o: o["forward_m"])
        for obstacle in obstacles:
            forward = obstacle["forward_m"] - speed * age * 0.2
            road_left = obstacle["road_lateral_m"]
            margin = 1.4 if self.version == 1 else 1.8
            clearance = 1.65 if self.version == 1 else 1.75
            minimum_offset = 0.3 if self.version == 1 else 0.2
            if -1.5 < forward < 24 and abs(road_left) < margin:
                desired = (-1 if road_left >= 0 else 1) * min(
                    0.8, max(minimum_offset, clearance - abs(road_left))
                )
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
            sensor_speed = 6.0 if self.version == 1 else 4.0 if self.version < 5 else 2.8
            desired_speed = min(desired_speed, sensor_speed)
        if abs(error) > 0.5:
            desired_speed = min(desired_speed, 4.0)
        if desired:
            avoidance_speed = 3.0 if self.version == 1 or abs(desired) > 0.35 else 6.0
            desired_speed = min(desired_speed, avoidance_speed)
        throttle = self.speed._lon.step(desired_speed - speed)
        action = np.array([steer, throttle], dtype=np.float32)
        if self.version >= 4:
            corrected = []
            for obstacle in info["obstacle_detections"]:
                x = obstacle["forward_m"] - shift_x
                y = obstacle["left_m"] - shift_y
                corrected.append(
                    {
                        "forward_m": float(x * np.cos(shift_yaw) + y * np.sin(shift_yaw)),
                        "left_m": float(-x * np.sin(shift_yaw) + y * np.cos(shift_yaw)),
                    }
                )
            action, self.last_decision = filter_action(
                action, speed, corrected, lateral, heading, curvature, self.road_half_width_m
            )
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
        if identity["name"] in {
            "robust",
            "robust-v1",
            "robust-v2",
            "robust-v3",
            "robust-v4",
            "robust-v5",
        }:
            version = {
                "robust-v1": 1,
                "robust-v2": 2,
                "robust-v3": 3,
                "robust-v4": 4,
                "robust-v5": 5,
                "robust": 6,
            }[identity["name"]]
            return RobustController(scenario, version=version)
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
