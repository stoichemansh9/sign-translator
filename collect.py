import os, sys
import cv2
import numpy as np
from utils import make_detector, extract

# Usage: python collect.py doubt
if len(sys.argv) < 2:
    print("Usage: python collect.py <sign_name>   e.g. python collect.py hello")
    sys.exit(1)
LABEL = sys.argv[1]
SAMPLES = 150
os.makedirs("data", exist_ok=True)

detector = make_detector()
cap = cv2.VideoCapture(0)
X, recording = [], False
print("SPACE = record on/off | Q = quit")

while len(X) < SAMPLES:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    feat, _ = extract(frame, detector)
    if feat is not None and recording:
        X.append(feat)

    status = "REC" if recording else "paused"
    cv2.putText(frame, f"{LABEL}: {len(X)}/{SAMPLES} [{status}]",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("collect", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        recording = not recording
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
np.save(f"data/{LABEL}.npy", np.array(X))
print(f"Saved {len(X)} samples for '{LABEL}'")