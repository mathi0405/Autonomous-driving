"""Exercise asynchronous camera handling without requiring a CARLA installation."""

import queue

import numpy as np
import pytest

from ad_rl.envs.sensors import SensorManager


def manager():
    result = object.__new__(SensorManager)
    result._image_queue = queue.Queue()
    result.latest_frame = None
    result.latest_image = None
    return result


def test_camera_discards_stale_frames_and_waits_for_exact_frame():
    sensors = manager()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    sensors._image_queue.put((7, image + 1))
    sensors._image_queue.put((8, image + 2))
    result = sensors.wait_for_frame(8, 0.1)
    assert np.all(result == 2) and sensors.latest_frame == 8


def test_future_frame_is_not_silently_used():
    sensors = manager()
    sensors._image_queue.put((9, np.zeros((2, 2, 3))))
    with pytest.raises(RuntimeError, match="skipped frame"):
        sensors.wait_for_frame(8, 0.1)


def test_missing_frame_has_a_bounded_timeout():
    with pytest.raises(TimeoutError, match="frame 8"):
        manager().wait_for_frame(8, 0.01)


def test_collision_flag_remains_latched():
    sensors = manager()
    sensors._collision_flag = True
    assert sensors.had_collision() and sensors.had_collision()
