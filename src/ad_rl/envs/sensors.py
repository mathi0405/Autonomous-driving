"""CARLA sensor management: RGB camera, collision and lane-invasion sensors.

This module is only imported when actually constructing a :class:`CarlaEnv`, so
it can safely ``import carla`` at call time. None of it is needed for the
fallback environment, the tests, or CI.
"""

from __future__ import annotations

import contextlib
import weakref
from typing import Any, List, Optional, Tuple

import numpy as np


class SensorManager:
    """Attaches and reads from the ego vehicle's sensor suite.

    Sensors used
    ------------
    * ``sensor.camera.rgb``        -- front-facing camera (policy observation).
    * ``sensor.other.collision``   -- collision events (reward + termination).
    * ``sensor.other.lane_invasion`` -- lane-marking crossings (infraction metric).

    The class deliberately keeps only the *latest* frame / event so the env can
    poll synchronously each tick.
    """

    def __init__(self, world: Any, vehicle: Any, image_size: Tuple[int, int] = (84, 84)) -> None:
        self._world = world
        self._vehicle = vehicle
        self._img_h, self._img_w = image_size
        self._sensors: List[Any] = []

        self.latest_image: np.ndarray = np.zeros((self._img_h, self._img_w, 3), dtype=np.uint8)
        self.collision_intensity: float = 0.0
        self.lane_invasions: int = 0
        self._collision_flag = False

        self._spawn_sensors()

    def _spawn_sensors(self) -> None:
        import carla  # local import: only needed with a live server

        bp_lib = self._world.get_blueprint_library()
        weak_self = weakref.ref(self)

        # RGB camera mounted on the hood.
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self._img_w))
        cam_bp.set_attribute("image_size_y", str(self._img_h))
        cam_bp.set_attribute("fov", "90")
        cam_tf = carla.Transform(carla.Location(x=1.6, z=1.7))
        camera = self._world.spawn_actor(cam_bp, cam_tf, attach_to=self._vehicle)
        camera.listen(lambda image: SensorManager._on_image(weak_self, image))
        self._sensors.append(camera)

        # Collision sensor.
        col_bp = bp_lib.find("sensor.other.collision")
        collision = self._world.spawn_actor(col_bp, carla.Transform(), attach_to=self._vehicle)
        collision.listen(lambda event: SensorManager._on_collision(weak_self, event))
        self._sensors.append(collision)

        # Lane-invasion sensor.
        lane_bp = bp_lib.find("sensor.other.lane_invasion")
        lane = self._world.spawn_actor(lane_bp, carla.Transform(), attach_to=self._vehicle)
        lane.listen(lambda event: SensorManager._on_lane_invasion(weak_self, event))
        self._sensors.append(lane)

    # ----- sensor callbacks (static to avoid holding strong refs) ---------- #
    @staticmethod
    def _on_image(weak_self: "weakref.ref[SensorManager]", image: Any) -> None:
        self = weak_self()
        if self is None:
            return
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]  # BGRA -> BGR
        self.latest_image = array[:, :, ::-1].copy()  # BGR -> RGB

    @staticmethod
    def _on_collision(weak_self: "weakref.ref[SensorManager]", event: Any) -> None:
        self = weak_self()
        if self is None:
            return
        impulse = event.normal_impulse
        self.collision_intensity = float(
            (impulse.x**2 + impulse.y**2 + impulse.z**2) ** 0.5
        )
        self._collision_flag = True

    @staticmethod
    def _on_lane_invasion(weak_self: "weakref.ref[SensorManager]", event: Any) -> None:
        self = weak_self()
        if self is None:
            return
        self.lane_invasions += len(event.crossed_lane_markings)

    # ----- polling helpers ------------------------------------------------- #
    def had_collision(self) -> bool:
        """Return and clear the collision flag for this tick."""
        flag = self._collision_flag
        self._collision_flag = False
        return flag

    def destroy(self) -> None:
        """Stop and destroy attached sensors, suppressing cleanup errors.

        The method intentionally suppresses exceptions raised during cleanup to
        ensure shutdown code does not fail; this mirrors prior behaviour but
        satisfies linter rules by using contextlib.suppress.
        """
        for sensor in self._sensors:
            # Stop/destroy each sensor and ignore any exceptions during cleanup.
            with contextlib.suppress(Exception):  # pragma: no cover - cleanup must never raise
                sensor.stop()
                sensor.destroy()
        self._sensors.clear()


__all__ = ["SensorManager"]
