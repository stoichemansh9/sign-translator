import time
import cv2
import joblib
from utils import make_detector, extract

clf = joblib.load("model.joblib")
detector = make_detector()
cap = cv2.VideoCapture(0)

times = []
while len(times) < 200:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    t0 = time.perf_counter()
    feat, _ = extract(frame, detector)
    if feat is not None:
        clf.predict_proba([feat])
    times.append(time.perf_counter() - t0)
    cv2.imshow("bench", frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

avg_ms = sum(times) / len(times) * 1000
print(f"Average latency: {avg_ms:.1f} ms | FPS: {1000 / avg_ms:.1f}")