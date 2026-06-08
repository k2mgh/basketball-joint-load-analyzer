def classify_dribble(left_wrist, right_wrist, left_hip, right_hip):
    """
    단순 규칙 기반 드리블 분류 MVP.
    실제 공 위치 추적 전까지는 손목 위치를 이용해 대략 분류.
    """

    body_center_x = (left_hip[0] + right_hip[0]) / 2
    left_wrist_x = left_wrist[0]
    right_wrist_x = right_wrist[0]

    if left_wrist_x < body_center_x and right_wrist_x > body_center_x:
        return "Crossover-like"

    if left_wrist_x > body_center_x and right_wrist_x < body_center_x:
        return "Crossover-like"

    if abs(left_wrist_x - right_wrist_x) < 0.08:
        return "Front Dribble"

    return "Unknown"