"""
AirSim utility functions for data generation.

Provides helpers for image format conversion, quaternion math, and file I/O
used by the orbital data collection pipeline.
"""

import re
import sys
import logging

import numpy as np


def string_to_uint8_array(bstr: bytes) -> np.ndarray:
    """Convert a binary string to a uint8 numpy array."""
    return np.frombuffer(bstr, np.uint8)


def string_to_float_array(bstr: bytes) -> np.ndarray:
    """Convert a binary string to a float32 numpy array."""
    return np.frombuffer(bstr, np.float32)


def list_to_2d_float_array(
    flst: list[float], width: int, height: int
) -> np.ndarray:
    """Reshape a flat float list into a 2D array of shape (height, width)."""
    return np.reshape(np.asarray(flst, np.float32), (height, width))


def get_pfm_array(response) -> np.ndarray:
    """Extract a 2D float array from an AirSim image response."""
    return list_to_2d_float_array(
        response.image_data_float, response.width, response.height
    )


def to_eularian_angles(q) -> tuple[float, float, float]:
    """Convert a quaternion to Euler angles (pitch, roll, yaw).

    Reference: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

    Args:
        q: Quaternion with x_val, y_val, z_val, w_val attributes.

    Returns:
        Tuple of (pitch, roll, yaw) in radians.
    """
    z = q.z_val
    y = q.y_val
    x = q.x_val
    w = q.w_val
    ysqr = y * y

    # roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    roll = np.arctan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    # yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    yaw = np.arctan2(t3, t4)

    return (pitch, roll, yaw)


def to_quaternion(pitch: float, roll: float, yaw: float):
    """Convert Euler angles to a quaternion.

    Args:
        pitch: Pitch angle in radians.
        roll: Roll angle in radians.
        yaw: Yaw angle in radians.

    Returns:
        Quaternion object with w_val, x_val, y_val, z_val attributes.
    """
    import math

    t0 = math.cos(yaw * 0.5)
    t1 = math.sin(yaw * 0.5)
    t2 = math.cos(roll * 0.5)
    t3 = math.sin(roll * 0.5)
    t4 = math.cos(pitch * 0.5)
    t5 = math.sin(pitch * 0.5)

    # Use a simple namespace for the quaternion
    class Quaternion:
        pass

    q = Quaternion()
    q.w_val = t0 * t2 * t4 + t1 * t3 * t5
    q.x_val = t0 * t3 * t4 - t1 * t2 * t5
    q.y_val = t0 * t2 * t5 + t1 * t3 * t4
    q.z_val = t1 * t2 * t4 - t0 * t3 * t5
    return q


def write_file(filename: str, bstr: bytes) -> None:
    """Write binary data to file (e.g., compressed PNG images)."""
    with open(filename, "wb") as f:
        f.write(bstr)


def read_pfm(file_path: str) -> tuple[np.ndarray, float]:
    """Read a PFM (Portable Float Map) depth image file.

    Args:
        file_path: Path to the .pfm file.

    Returns:
        Tuple of (data array, scale factor).
    """
    with open(file_path, "rb") as f:
        header = f.readline().rstrip()
        header = str(bytes.decode(header, encoding="utf-8"))
        if header == "PF":
            color = True
        elif header == "Pf":
            color = False
        else:
            raise ValueError("Not a PFM file.")

        temp_str = str(bytes.decode(f.readline(), encoding="utf-8"))
        dim_match = re.match(r"^(\d+)\s(\d+)\s$", temp_str)
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise ValueError("Malformed PFM header.")

        scale = float(f.readline().rstrip())
        if scale < 0:
            endian = "<"
            scale = -scale
        else:
            endian = ">"

        data = np.fromfile(f, endian + "f")
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)

    return data, scale


def write_pfm(file_path: str, image: np.ndarray, scale: float = 1) -> None:
    """Write a float32 numpy array to PFM format.

    Args:
        file_path: Output .pfm file path.
        image: Float32 numpy array (H x W or H x W x 3).
        scale: Scale factor (negative for little-endian).
    """
    if image.dtype.name != "float32":
        raise ValueError("Image dtype must be float32.")

    if len(image.shape) == 3 and image.shape[2] == 3:
        color = True
    elif len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        color = False
    else:
        raise ValueError("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

    with open(file_path, "wb") as f:
        f.write("PF\n".encode("utf-8") if color else "Pf\n".encode("utf-8"))
        f.write(f"{image.shape[1]} {image.shape[0]}\n".encode("utf-8"))

        endian = image.dtype.byteorder
        if endian == "<" or (endian == "=" and sys.byteorder == "little"):
            scale = -scale

        f.write(f"{scale}\n".encode("utf-8"))
        image.tofile(f)


def write_png(filename: str, image: np.ndarray) -> None:
    """Write a numpy array as a PNG image.

    Args:
        filename: Output PNG file path.
        image: Numpy array of shape H x W x C.
    """
    import cv2

    ret = cv2.imwrite(filename, image)
    if not ret:
        logging.error(f"Writing PNG file {filename} failed")
