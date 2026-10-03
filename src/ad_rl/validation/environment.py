"""Separate disturbed measurements from ground-truth evaluation telemetry."""

from __future__ import annotations

from collections import deque

import numpy as np

from ad_rl.envs.fallback_env import KinematicDrivingEnv
from ad_rl.utils.config import EnvConfig, FallbackConfig, RewardConfig
from ad_rl.validation.scenarios import Scenario


class ScenarioEnvironment(KinematicDrivingEnv):
    """Instrumented kinematic environment; fault-free telemetry remains independent."""

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        super().__init__(
            EnvConfig(
                observation="state",
                frame_stack=1,
                action_repeat=2,
                max_episode_steps=scenario.max_steps,
            ),
            RewardConfig(target_speed_kmh=scenario.target_speed_kmh),
            FallbackConfig(
                curviness=scenario.curviness,
                road_half_width_m=scenario.road_half_width_m,
                num_obstacles=scenario.num_obstacles,
            ),
            render_mode="rgb_array",
        )
        self._measurement_history: deque = deque(maxlen=scenario.observation_delay_steps + 1)
        self._noise_rng = np.random.default_rng(scenario.seed ^ 0xA5A5A5A5)

    def reset(self, *, seed=None, options=None):
        """Reset geometry and an independent sensor-noise random stream."""
        seed = self.scenario.seed if seed is None else seed
        super().reset(seed=seed, options=options)
        self._noise_rng = np.random.default_rng(seed ^ 0xA5A5A5A5)
        normal = np.array([-np.sin(self._yaw), np.cos(self._yaw)])
        self._x += float(normal[0] * self.scenario.initial_offset_m)
        self._y += float(normal[1] * self.scenario.initial_offset_m)
        if self.scenario.obstacle_offset_m is not None:
            indices = np.linspace(30, 100, self.scenario.num_obstacles, dtype=int)
            normals = np.column_stack(
                (-np.sin(self._path_yaw[indices]), np.cos(self._path_yaw[indices]))
            )
            self._obstacles = self._path_xy[indices] + self.scenario.obstacle_offset_m * normals
        lateral, heading = self._lateral_and_heading_error(0)
        self._measurement_history.clear()
        obs = self._build_observation(lateral, heading)
        info = self._info(lateral, heading, {})
        info.update(
            collision=False,
            offroad=False,
            is_success=False,
            route_fraction=0.0,
            terminal_reason="RUNNING",
        )
        info.update(self._sensing())
        return obs, info

    def step(self, action):
        """Apply an actuator bias without modifying the requested-action record."""
        action = np.asarray(action, dtype=np.float32).copy()
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Policy must produce two finite action components.")
        action[0] += self.scenario.steering_bias
        result = super().step(action)
        result[4]["applied_action"] = np.clip(action, -1, 1).tolist()
        result[4].update(self._sensing())
        return result

    def _sensing(self) -> dict:
        """Simulate local range detections, not unrestricted obstacle-map access."""
        detections = []
        for obstacle in self._obstacles:
            delta = obstacle - [self._x, self._y]
            forward = float(delta[0] * np.cos(self._yaw) + delta[1] * np.sin(self._yaw))
            left = float(-delta[0] * np.sin(self._yaw) + delta[1] * np.cos(self._yaw))
            if -4 < forward < 30 and abs(left) < 8:
                index = int(np.argmin(np.sum((self._path_xy - obstacle) ** 2, axis=1)))
                normal = np.array([-np.sin(self._path_yaw[index]), np.cos(self._path_yaw[index])])
                road_left = float(np.dot(obstacle - self._path_xy[index], normal))
                detections.append(
                    {"forward_m": forward, "left_m": left, "road_lateral_m": road_left}
                )
        return {"measurement_step": self._steps, "obstacle_detections": detections}

    def measured(self, obs: np.ndarray, info: dict) -> tuple[np.ndarray, dict]:
        """Return noisy/delayed policy inputs; never alter scoring inputs."""
        keys = (
            "speed_ms",
            "lateral_error_m",
            "heading_error_rad",
            "curvature_inv_m",
            "measurement_step",
            "obstacle_detections",
        )
        measured_info = {key: info[key] for key in keys}
        noise = float(self._noise_rng.normal(0, self.scenario.sensor_noise_std_m))
        lateral = info["lateral_error_m"] + self.scenario.lateral_bias_m + noise
        measured_info["lateral_error_m"] = lateral
        measured_obs = obs.copy()
        measured_obs[1] = lateral / self.scenario.road_half_width_m
        self._measurement_history.append((measured_obs, measured_info))
        return self._measurement_history[0]

    def scene(self) -> dict:
        """Export actual generated geometry for a standalone replay viewer."""
        return {
            "road": self._path_xy.tolist(),
            "obstacles": self._obstacles.tolist(),
            "road_half_width_m": self.scenario.road_half_width_m,
            "collision_radius_m": 1.2,
        }
