"""Camera field-of-view checks and pinhole measurement modeling.

Part G supplies visibility, projection, and pixel covariance (docs/HUONG_DAN_KY_THUAT.md §2).
The platform differentiates projection using a chain-rule Jacobian.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np

from fusion_lab.workspace_support import get_tracking_params

Matrix = np.matrix | np.ndarray

MIN_DEPTH = 1e-6


def _to_sensor_frame(x: Matrix, sensor: Any) -> np.ndarray:
    """Return the position part of state x in sensor axes (forward, left, up)."""
    # vi: p_s = R @ p + t, với [R | t] lấy từ ma trận 4x4 veh_to_sens.
    position = np.asarray(x, dtype=float).reshape(-1)[:3]
    transform = np.asarray(sensor.veh_to_sens, dtype=float)
    return transform[:3, :3] @ position + transform[:3, 3]


def is_in_field_of_view(x: Matrix, sensor: Any) -> bool:
    """Return True if state x is visible within the sensor horizontal field of view.

    Args:
        x: State vector (6x1) with position in vehicle frame.
        sensor: Lidar or camera adapter with ``veh_to_sens`` and ``fov``
            (radians).

    Returns:
        True if sensor coordinates are finite and the horizontal angle is within
        ``sensor.fov``. A camera additionally requires depth > 1e-6.
    """
    p_s = _to_sensor_frame(x, sensor)
    if not np.isfinite(p_s).all():
        return False
    forward, left = p_s[0], p_s[1]
    # vi: Camera không chiếu được điểm nằm sau hoặc ngay trên mặt phẳng ảnh.
    if getattr(sensor, "name", None) == "camera" and forward <= MIN_DEPTH:
        return False
    # vi: Góc ngang so với trục nhìn; sensor.fov là (góc nhỏ nhất, góc lớn nhất).
    angle = np.arctan2(left, forward)
    lower, upper = sensor.fov
    return bool(lower <= angle <= upper)


def camera_measurement_prediction(x: Matrix, sensor: Any) -> Matrix:
    """Predict image-plane measurement h(x) using the pinhole camera model.

    Args:
        x: State vector.
        sensor: Camera with intrinsics ``f_i, f_j, c_i, c_j``.

    Returns:
        2x1 predicted pixel coordinates as ``np.matrix``.

    Raises:
        ValueError: With coordinate context if sensor coordinates are nonfinite
            or depth is at most 1e-6.
    """
    p_s = _to_sensor_frame(x, sensor)
    forward, left, up = p_s
    # vi: Chặn trước phép chia — toạ độ không hữu hạn hoặc độ sâu không dương.
    if not np.isfinite(p_s).all() or forward <= MIN_DEPTH:
        raise ValueError(
            "Camera projection needs finite coordinates and depth "
            f"> {MIN_DEPTH:g}; sensor position={p_s.tolist()}"
        )
    # vi: Trục ảnh ngược chiều trục trái/lên của camera nên mang dấu trừ.
    u = sensor.c_i - sensor.f_i * left / forward
    v = sensor.c_j - sensor.f_j * up / forward
    return np.asmatrix([[u], [v]])


def build_camera_measurement(z: Sequence[float], sensor: Any) -> dict[str, Any]:
    """Build camera measurement vector z and covariance R from pixel coordinates.

    Args:
        z: Sequence ``[u, v]`` pixel coordinates.
        sensor: Camera sensor object.

    Returns:
        Dict with keys ``z``, ``R``, ``sensor``.
    """
    params = get_tracking_params()
    pixels = np.asarray(z, dtype=float).reshape(-1)[:2]
    # vi: Nhiễu hai trục ảnh độc lập, đơn vị pixel.
    R = np.asmatrix(np.diag([params.sigma_cam_i**2, params.sigma_cam_j**2]))
    return {"z": np.asmatrix(pixels).T, "R": R, "sensor": sensor}
