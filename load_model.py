import math


def clamp(x, low=0, high=100):
    return max(low, min(high, x))


def normalize(value, warning_value, critical_value):
    """
    value가 warning_value보다 작으면 거의 위험하지 않음.
    value가 critical_value에 가까워질수록 100에 가까워짐.
    """
    if value <= warning_value:
        return 0

    if value >= critical_value:
        return 100

    return (value - warning_value) / (critical_value - warning_value) * 100


def smooth_score(raw_score):
    """
    점수가 너무 쉽게 100이 되는 것을 방지.
    실제로 매우 큰 위험이 아니면 100에 도달하지 않도록 완만하게 압축.
    """
    raw_score = clamp(raw_score)

    # 0~100을 더 완만하게 변환
    return 100 * (1 - math.exp(-raw_score / 55))


def angle_risk(angle_value, safe_min, safe_max):
    if angle_value is None:
        return 0

    if safe_min <= angle_value <= safe_max:
        return 0

    if angle_value < safe_min:
        deviation = safe_min - angle_value
    else:
        deviation = angle_value - safe_max

    # 10도 이탈부터 주의, 70도 이상 이탈이면 매우 위험
    return normalize(
        deviation,
        warning_value=10,
        critical_value=70
    )


def velocity_risk(velocity):
    """
    각속도 기반 위험도.
    기존에는 velocity가 그대로 점수에 들어가서 100이 너무 쉽게 나왔음.
    """
    return normalize(
        velocity,
        warning_value=120,
        critical_value=900
    )


def repeat_risk(repeat_factor):
    """
    반복 피로도.
    현재 repeat_factor가 1이면 거의 영향 없음.
    """
    return normalize(
        repeat_factor,
        warning_value=10,
        critical_value=100
    )


def joint_load_score(angle_value, velocity, repeat_factor, safe_min, safe_max):
    a_risk = angle_risk(angle_value, safe_min, safe_max)
    v_risk = velocity_risk(velocity)
    r_risk = repeat_risk(repeat_factor)

    raw_score = (
        0.55 * a_risk +
        0.35 * v_risk +
        0.10 * r_risk
    )

    return clamp(smooth_score(raw_score))


def total_risk_level(score):
    if score >= 85:
        return "HIGH"
    elif score >= 60:
        return "MEDIUM"
    else:
        return "LOW"


def feedback(joint_name, score):
    level = total_risk_level(score)

    if level == "HIGH":
        return f"{joint_name}: 위험도가 높습니다. 해당 구간의 자세를 다시 확인하는 것이 좋습니다."
    elif level == "MEDIUM":
        return f"{joint_name}: 중간 수준의 부하가 있습니다. 반복되면 피로가 누적될 수 있습니다."
    else:
        return f"{joint_name}: 현재 부하는 낮은 편입니다."