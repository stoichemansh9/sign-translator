import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands

def make_detector():
    return mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                          min_detection_confidence=0.7,
                          min_tracking_confidence=0.5)

def normalize(landmarks):
    """21 landmarks -> 63 features. Wrist is the origin, scaled by hand size."""
    pts = np.array([[p.x, p.y, p.z] for p in landmarks])
    pts = pts - pts[0]                       # wrist as origin (position-invariant)
    scale = np.linalg.norm(pts[9]) + 1e-6    # wrist -> middle finger base (size-invariant)
    return (pts / scale).flatten()

def extract(frame, detector):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = detector.process(rgb)
    if not res.multi_hand_landmarks:
        return None, res
    return normalize(res.multi_hand_landmarks[0].landmark), res