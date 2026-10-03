"""Local kinematic prediction for an explicitly measured controller safety filter."""

from __future__ import annotations

import math

import numpy as np


def predict_clearance(
    steer: float, speed: float, obstacles: list[dict], horizon: float = 2.0
) -> float:
    """Predict circular obstacle clearance under constant speed and steering.

    Obstacles are local sensor detections, not the simulator's global obstacle
    array. Prediction is approximate; passing this filter is not a guarantee.
    """
    if not obstacles:
        return 100.0
    xy = np.array([[o["forward_m"], o["left_m"]] for o in obstacles])
    x, y, yaw = 0.0, 0.0, 0.0
    minimum = float(np.min(np.linalg.norm(xy, axis=1))) - 1.2
    for _ in range(round(horizon / 0.1)):
        x += speed * math.cos(yaw) * 0.1
        y += speed * math.sin(yaw) * 0.1
        yaw += speed / 2.8 * math.tan(steer * 0.5) * 0.1
        minimum = min(minimum, float(np.min(np.linalg.norm(xy - [x, y], axis=1))) - 1.2)
    return minimum


def filter_action(
    action: np.ndarray,
    speed: float,
    obstacles: list[dict],
    lateral: float,
    heading: float,
    curvature: float,
    road_half_width: float,
) -> tuple[np.ndarray, dict]:
    """Choose a nearby steering command with predicted clearance, or brake.

    The road-envelope check uses measured lane offset and constant curvature,
    with a 0.15 m margin. Collision prediction uses a 0.2 m obstacle margin.
    """
    relevant = [o for o in obstacles if -2 < o["forward_m"] < max(6, speed * 2.5)]
    requested = action.copy()
    current_clearance = min(
        (math.hypot(o["forward_m"], o["left_m"]) - 1.2 for o in relevant), default=100.0
    )
    required_clearance = max(0.0, min(0.2, current_clearance))
    initial = predict_clearance(float(action[0]), speed, relevant)

    def stays_in_lane(steer: float) -> bool:
        angle, error = heading, lateral
        for _ in range(20):
            error += speed * math.sin(angle) * 0.1
            angle += (speed / 2.8 * math.tan(steer * 0.5) - speed * curvature) * 0.1
            if abs(error) > road_half_width - 0.15:
                return False
        return True

    if initial >= required_clearance - 1e-9 and stays_in_lane(float(action[0])):
        return action, {"intervened": False, "predicted_clearance_m": initial}
    safe = []
    for steer in np.linspace(-1.0, 1.0, 21):
        in_lane = stays_in_lane(float(steer))
        clearance = predict_clearance(float(steer), speed, relevant)
        if in_lane and clearance >= required_clearance - 1e-9:
            safe.append((abs(float(steer) - float(action[0])), float(steer), clearance))
    if safe:
        _, steer, clearance = min(safe)
        action = np.array([steer, min(float(action[1]), 0.0)], dtype=np.float32)
        reason = "STEERING_CLEARANCE"
    else:
        action = np.array([float(action[0]), -1.0], dtype=np.float32)
        clearance = initial
        reason = "NO_CLEAR_PREDICTION_BRAKE"
    return action, {
        "intervened": True,
        "reason": reason,
        "raw_action": requested.tolist(),
        "predicted_clearance_m": clearance,
    }
