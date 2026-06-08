import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def create_pose_model():
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


def extract_pose(frame, pose_model):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pose_model.process(rgb)


def draw_pose(frame, result):
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
    return frame