import os
import cv2
import pandas as pd

from pose_utils import create_pose_model, extract_pose
from biomechanics import point, angle, angular_velocity, asymmetry
from load_model import joint_load_score
from dribble_classifier import classify_dribble
from feedback_report import generate_report

from config import *

def set_progress(progress_callback, percent, message):
    if progress_callback:
        progress_callback(percent, message)

def get_fps(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        return 30
    return fps


def analyze_video(video_path, output_csv_path, output_report_path, progress_callback=None):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Video file cannot be opened.")

    fps = get_fps(cap)
    pose_model = create_pose_model()

    prev = {
        "left_knee": None,
        "right_knee": None,
        "left_hip": None,
        "right_hip": None,
        "left_elbow": None,
        "right_elbow": None,
        "left_ankle": None,
        "right_ankle": None,
    }

    rows = []
    frame_idx = 0
    repeat_factor = 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    set_progress(progress_callback, 0, "분석을 시작했습니다.")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        result = extract_pose(frame, pose_model)

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark

            left_shoulder = point(lm, LEFT_SHOULDER)
            right_shoulder = point(lm, RIGHT_SHOULDER)

            left_elbow = point(lm, LEFT_ELBOW)
            right_elbow = point(lm, RIGHT_ELBOW)

            left_wrist = point(lm, LEFT_WRIST)
            right_wrist = point(lm, RIGHT_WRIST)

            left_hip = point(lm, LEFT_HIP)
            right_hip = point(lm, RIGHT_HIP)

            left_knee = point(lm, LEFT_KNEE)
            right_knee = point(lm, RIGHT_KNEE)

            left_ankle = point(lm, LEFT_ANKLE)
            right_ankle = point(lm, RIGHT_ANKLE)

            left_foot = point(lm, LEFT_FOOT)
            right_foot = point(lm, RIGHT_FOOT)

            left_knee_angle = angle(left_hip, left_knee, left_ankle)
            right_knee_angle = angle(right_hip, right_knee, right_ankle)

            left_hip_angle = angle(left_shoulder, left_hip, left_knee)
            right_hip_angle = angle(right_shoulder, right_hip, right_knee)

            left_elbow_angle = angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = angle(right_shoulder, right_elbow, right_wrist)

            left_ankle_angle = angle(left_knee, left_ankle, left_foot)
            right_ankle_angle = angle(right_knee, right_ankle, right_foot)

            left_knee_vel = angular_velocity(left_knee_angle, prev["left_knee"], fps)
            right_knee_vel = angular_velocity(right_knee_angle, prev["right_knee"], fps)

            left_hip_vel = angular_velocity(left_hip_angle, prev["left_hip"], fps)
            right_hip_vel = angular_velocity(right_hip_angle, prev["right_hip"], fps)

            left_elbow_vel = angular_velocity(left_elbow_angle, prev["left_elbow"], fps)
            right_elbow_vel = angular_velocity(right_elbow_angle, prev["right_elbow"], fps)

            left_ankle_vel = angular_velocity(left_ankle_angle, prev["left_ankle"], fps)
            right_ankle_vel = angular_velocity(right_ankle_angle, prev["right_ankle"], fps)

            knee_asymmetry = asymmetry(left_knee_angle, right_knee_angle)
            hip_asymmetry = asymmetry(left_hip_angle, right_hip_angle)
            ankle_asymmetry = asymmetry(left_ankle_angle, right_ankle_angle)
            elbow_asymmetry = asymmetry(left_elbow_angle, right_elbow_angle)

            left_knee_load = joint_load_score(left_knee_angle, left_knee_vel, repeat_factor, 90, 170)
            right_knee_load = joint_load_score(right_knee_angle, right_knee_vel, repeat_factor, 90, 170)

            left_hip_load = joint_load_score(left_hip_angle, left_hip_vel, repeat_factor, 70, 170)
            right_hip_load = joint_load_score(right_hip_angle, right_hip_vel, repeat_factor, 70, 170)

            left_elbow_load = joint_load_score(left_elbow_angle, left_elbow_vel, repeat_factor, 70, 180)
            right_elbow_load = joint_load_score(right_elbow_angle, right_elbow_vel, repeat_factor, 70, 180)

            left_ankle_load = joint_load_score(left_ankle_angle, left_ankle_vel, repeat_factor, 70, 130)
            right_ankle_load = joint_load_score(right_ankle_angle, right_ankle_vel, repeat_factor, 70, 130)

            dribble_type = classify_dribble(left_wrist, right_wrist, left_hip, right_hip)

            load_dict = {
                "left_knee_load": left_knee_load,
                "right_knee_load": right_knee_load,
                "left_hip_load": left_hip_load,
                "right_hip_load": right_hip_load,
                "left_elbow_load": left_elbow_load,
                "right_elbow_load": right_elbow_load,
                "left_ankle_load": left_ankle_load,
                "right_ankle_load": right_ankle_load,
            }

            max_load = max(load_dict.values())
            max_load_joint = max(load_dict, key=load_dict.get)

            rows.append({
                "frame": frame_idx,
                "time_sec": frame_idx / fps,
                "dribble_type": dribble_type,

                "left_knee_angle": left_knee_angle,
                "right_knee_angle": right_knee_angle,
                "left_hip_angle": left_hip_angle,
                "right_hip_angle": right_hip_angle,
                "left_elbow_angle": left_elbow_angle,
                "right_elbow_angle": right_elbow_angle,
                "left_ankle_angle": left_ankle_angle,
                "right_ankle_angle": right_ankle_angle,

                "left_knee_velocity": left_knee_vel,
                "right_knee_velocity": right_knee_vel,
                "left_hip_velocity": left_hip_vel,
                "right_hip_velocity": right_hip_vel,
                "left_elbow_velocity": left_elbow_vel,
                "right_elbow_velocity": right_elbow_vel,
                "left_ankle_velocity": left_ankle_vel,
                "right_ankle_velocity": right_ankle_vel,

                "left_knee_load": left_knee_load,
                "right_knee_load": right_knee_load,
                "left_hip_load": left_hip_load,
                "right_hip_load": right_hip_load,
                "left_elbow_load": left_elbow_load,
                "right_elbow_load": right_elbow_load,
                "left_ankle_load": left_ankle_load,
                "right_ankle_load": right_ankle_load,

                "knee_asymmetry": knee_asymmetry,
                "hip_asymmetry": hip_asymmetry,
                "ankle_asymmetry": ankle_asymmetry,
                "elbow_asymmetry": elbow_asymmetry,

                "max_load": max_load,
                "max_load_joint": max_load_joint,
            })

            prev["left_knee"] = left_knee_angle
            prev["right_knee"] = right_knee_angle
            prev["left_hip"] = left_hip_angle
            prev["right_hip"] = right_hip_angle
            prev["left_elbow"] = left_elbow_angle
            prev["right_elbow"] = right_elbow_angle
            prev["left_ankle"] = left_ankle_angle
            prev["right_ankle"] = right_ankle_angle

        frame_idx += 1

        if total_frames > 0 and frame_idx % 10 == 0:
            percent = int(frame_idx / total_frames * 100)
            set_progress(progress_callback, min(percent, 99), f"영상 분석 중... {percent}%")

    cap.release()

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise ValueError("No pose data was detected. Try using a clearer full-body video.")

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)

    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    set_progress(progress_callback, 99, "리포트를 생성하는 중입니다.")

    report = generate_report(df, fps)

    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write(report)

    summary = make_summary(df)

    set_progress(progress_callback, 100, "분석이 완료되었습니다.")

    return {
        "report": report,
        "summary": summary,
        "csv_path": output_csv_path,
        "report_path": output_report_path,
    }


def make_summary(df):
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
    peak_joint = max_loads.idxmax()

    return {
        "most_loaded_joint": most_loaded_joint,
        "most_loaded_score": round(float(avg_loads[most_loaded_joint]), 1),
        "peak_joint": peak_joint,
        "peak_score": round(float(max_loads[peak_joint]), 1),
        "avg_knee_asymmetry": round(float(df["knee_asymmetry"].mean()), 1),
        "max_load": round(float(df["max_load"].max()), 1),
    }