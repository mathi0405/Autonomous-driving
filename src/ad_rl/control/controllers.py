"""Classical driving controllers used as a non-learning baseline.

Why include these? An internship reviewer wants to see that the RL policy is
actually *worth it*. A tuned PID + Stanley controller is the standard yardstick
in autonomous driving: if the learned policy can't beat (or at least match) a
hand-built controller on the same route, that tells you something. The same
``VehiclePIDController`` drives both the fallback env and CARLA, because both
expose ``speed_ms``, ``lateral_error_m`` and ``heading_error_rad`` in their step
``info`` dict.

Sign conventions (consistent with the environments)
---------------------------------------------------
* ``steer`` in ``[-1, 1]``; positive steers left (increases yaw).
* ``lateral_error_m`` is the signed offset onto the road's left-normal -- a car
  displaced to the left has a positive value and must steer right (negative).
* ``heading_error_rad`` is ``yaw - road_yaw`` wrapped to ``[-pi, pi]``.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


class PID:
    """A small, clamped PID controller with anti-windup."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        dt: float = 0.1,
        integral_clip: float = 5.0,
        output_clip: Tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = dt
        self.integral_clip = integral_clip
        self.output_clip = output_clip
        self._integral = 0.0
        self._prev_error = 0.0

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0

    def step(self, error: float) -> float:
        """Advance the PID controller by one timestep.

        Parameters
        ----------
        error:
            The current error signal (setpoint - measurement).

        Returns
        -------
        float
            Clipped controller output in the configured output range.
        """
        self._integral = float(
            np.clip(self._integral + error * self.dt, -self.integral_clip, self.integral_clip)
        )
        derivative = (error - self._prev_error) / self.dt
        self._prev_error = error
        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return float(np.clip(output, *self.output_clip))


class VehiclePIDController:
    """Longitudinal PID (speed) + lateral PID/Stanley (cross-track) controller.

    Parameters
    ----------
    target_speed_ms:
        Desired cruising speed in m/s.
    dt:
        Control timestep (should match the env's effective step time).
    lateral:
        ``"pid"`` (cross-track + heading PID) or ``"stanley"`` (Stanley steering).
    """

    def __init__(
        self,
        target_speed_ms: float,
        dt: float = 0.1,
        lateral: str = "pid",
        stanley_gain: float = 4.0,
        max_steer_rad: float = 0.5,
    ) -> None:
        self.target_speed_ms = target_speed_ms
        self.lateral_mode = lateral
        self.stanley_gain = stanley_gain
        # Steering laws that produce a physical angle (Stanley) are normalised by
        # the vehicle's max steering angle to match the [-1, 1] action space.
        self.max_steer_rad = max_steer_rad
        self._lon = PID(kp=0.5, ki=0.05, kd=0.0, dt=dt, output_clip=(-1.0, 1.0))
        self._lat = PID(kp=0.6, ki=0.0, kd=0.3, dt=dt, output_clip=(-1.0, 1.0))

    def reset(self) -> None:
        self._lon.reset()
        self._lat.reset()

    def act(
        self, speed_ms: float, lateral_error_m: float, heading_error_rad: float
    ) -> np.ndarray:
        """Return a ``[steer, throttle_brake]`` action in ``[-1, 1]^2``."""
        # Longitudinal: PID on speed error.
        throttle_brake = self._lon.step(self.target_speed_ms - speed_ms)

        # Lateral: choose steering law.
        if self.lateral_mode == "stanley":
            # Stanley: align heading and correct cross-track, damped by speed.
            # Produces a steering *angle* (rad), normalised to the action space.
            cross_track_term = math.atan2(self.stanley_gain * (-lateral_error_m), speed_ms + 1.0)
            steer_rad = -heading_error_rad + cross_track_term
            steer = float(np.clip(steer_rad / self.max_steer_rad, -1.0, 1.0))
        else:
            steer = self._lat.step(-(lateral_error_m + 0.8 * heading_error_rad))

        return np.array([steer, throttle_brake], dtype=np.float32)

    def act_from_info(self, info: dict) -> np.ndarray:
        """Convenience: read the standard keys from an env ``info`` dict."""
        return self.act(
            speed_ms=float(info.get("speed_ms", 0.0)),
            lateral_error_m=float(info.get("lateral_error_m", 0.0)),
            heading_error_rad=float(info.get("heading_error_rad", 0.0)),
        )


__all__ = ["PID", "VehiclePIDController"]
