import numpy as np


def point(landmarks, idx):
    lm = landmarks[idx]
    return np.array([lm.x, lm.y])


def angle(a, b, c):
    ba = a - b
    bc = c - b

    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return None

    cos_val = np.dot(ba, bc) / denom
    cos_val = np.clip(cos_val, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_val)))


def angular_velocity(current_angle, previous_angle, fps):
    if current_angle is None or previous_angle is None:
        return 0.0
    return abs(current_angle - previous_angle) * fps


def asymmetry(left_value, right_value):
    if left_value is None or right_value is None:
        return 0.0
    return abs(left_value - right_value)