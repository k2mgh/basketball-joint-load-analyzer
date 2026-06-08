HIGH_THRESHOLD = 85
MEDIUM_THRESHOLD = 65
TOP_N_MOMENTS = 5


def risk_level(score):
    if score >= HIGH_THRESHOLD:
        return "HIGH"
    elif score >= MEDIUM_THRESHOLD:
        return "MEDIUM"
    else:
        return "LOW"


def format_time(seconds):
    seconds = int(round(seconds))
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def pretty_joint_name(joint):
    names = {
        "left_knee_load": "Left Knee",
        "right_knee_load": "Right Knee",
        "left_hip_load": "Left Hip",
        "right_hip_load": "Right Hip",
        "left_elbow_load": "Left Elbow",
        "right_elbow_load": "Right Elbow",
        "left_ankle_load": "Left Ankle",
        "right_ankle_load": "Right Ankle",
    }
    return names.get(joint, joint)


def generate_critical_moments(df, top_n=TOP_N_MOMENTS):
    if "time_sec" not in df.columns:
        df["time_sec"] = df["frame"]

    filtered = df[df["max_load"] >= MEDIUM_THRESHOLD].copy()

    if len(filtered) == 0:
        return []

    # 같은 1초 안에서 여러 프레임이 잡히는 문제 방지
    filtered["time_bucket"] = filtered["time_sec"].round().astype(int)

    moments = (
        filtered.sort_values("max_load", ascending=False)
        .drop_duplicates(subset=["time_bucket"])
        .head(top_n)
    )

    result = []

    for _, row in moments.iterrows():
        result.append({
            "time": format_time(row["time_sec"]),
            "joint": pretty_joint_name(row["max_load_joint"]),
            "score": round(float(row["max_load"]), 1),
            "level": risk_level(float(row["max_load"])),
        })

    return result


def generate_report(df, fps):
    load_cols = [
        "left_knee_load",
        "right_knee_load",
        "left_hip_load",
        "right_hip_load",
        "left_elbow_load",
        "right_elbow_load",
        "left_ankle_load",
        "right_ankle_load",
    ]

    avg_loads = df[load_cols].mean()
    max_loads = df[load_cols].max()

    most_loaded_joint = avg_loads.idxmax()
    highest_peak_joint = max_loads.idxmax()

    knee_asym_avg = df["knee_asymmetry"].mean()
    knee_asym_max = df["knee_asymmetry"].max()

    critical_moments = generate_critical_moments(df)

    report = []
    report.append("===== Basketball Joint Load Analysis Report =====")
    report.append("")

    report.append("[1] Summary")
    report.append(
        f"- 가장 평균 부하가 큰 관절: {pretty_joint_name(most_loaded_joint)} "
        f"({avg_loads[most_loaded_joint]:.1f}/100)"
    )
    report.append(
        f"- 순간 최대 부하가 큰 관절: {pretty_joint_name(highest_peak_joint)} "
        f"({max_loads[highest_peak_joint]:.1f}/100)"
    )
    report.append(f"- 평균 무릎 좌우 비대칭: {knee_asym_avg:.1f}°")
    report.append(f"- 최대 무릎 좌우 비대칭: {knee_asym_max:.1f}°")
    report.append("")

    report.append("[2] Average Joint Load")
    for col in load_cols:
        report.append(
            f"- {pretty_joint_name(col)}: {avg_loads[col]:.1f}/100 "
            f"({risk_level(avg_loads[col])})"
        )

    report.append("")
    report.append("[3] Critical Moments")

    if len(critical_moments) == 0:
        report.append("- MEDIUM 이상 위험 순간은 감지되지 않았습니다.")
    else:
        for i, moment in enumerate(critical_moments, start=1):
            report.append(
                f"{i}. {moment['time']} | {moment['joint']} | "
                f"Load Score: {moment['score']}/100 ({moment['level']})"
            )

    report.append("")
    report.append("[4] Main Feedback")

    if "knee" in most_loaded_joint:
        report.append(
            "- 무릎 부하가 가장 크게 나타났습니다. 드리블 중 무릎이 과도하게 굽혀지거나 "
            "방향 전환 시 무릎이 안쪽으로 무너지는 동작을 주의해야 합니다."
        )
    elif "ankle" in most_loaded_joint:
        report.append(
            "- 발목 부하가 가장 크게 나타났습니다. 급격한 방향 전환, 착지, 발목 회전이 "
            "반복되는 구간에서 부상 위험이 커질 수 있습니다."
        )
    elif "hip" in most_loaded_joint:
        report.append(
            "- 고관절 부하가 가장 크게 나타났습니다. 낮은 자세 유지와 상체 기울기가 "
            "반복되면서 고관절 주변 근육에 부담이 누적될 수 있습니다."
        )
    elif "elbow" in most_loaded_joint:
        report.append(
            "- 팔꿈치 부하가 가장 크게 나타났습니다. 반복적인 공 컨트롤 과정에서 "
            "팔꿈치 굴곡/신전이 많이 발생했을 가능성이 있습니다."
        )

    if knee_asym_avg >= 15:
        report.append(
            "- 좌우 무릎 각도 차이가 큽니다. 한쪽 다리에 체중이 더 많이 실리는 "
            "습관이 있을 가능성이 있습니다."
        )
    elif knee_asym_avg >= 8:
        report.append(
            "- 좌우 무릎 각도 차이가 약간 있습니다. 반복되면 한쪽 관절에 피로가 "
            "누적될 수 있습니다."
        )
    else:
        report.append("- 좌우 무릎 비대칭성은 비교적 낮은 편입니다.")

    if len(critical_moments) > 0:
        top = critical_moments[0]
        report.append(
            f"- 가장 위험했던 대표 순간은 {top['time']}이며, "
            f"{top['joint']}에서 {top['score']}/100의 부하가 관측되었습니다."
        )

    report.append("")
    report.append("[5] How to Use This Result")
    report.append(
        "- Critical Moments는 영상을 다시 확인할 때 우선적으로 살펴볼 대표 시점입니다."
    )
    report.append(
        "- 평균 부하가 높은 관절은 반복 훈련 시 피로가 누적될 가능성이 높은 부위로 해석합니다."
    )
    report.append(
        "- 순간 최대 부하가 높은 관절은 특정 동작 순간의 부상 위험과 관련이 있을 수 있습니다."
    )
    report.append(
        "- 본 결과는 실제 관절 토크를 직접 측정한 값이 아니라, 영상 기반 관절 각도와 "
        "각속도를 이용한 상대적 부하 추정값입니다."
    )

    return "\n".join(report)