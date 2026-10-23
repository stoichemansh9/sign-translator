import time
from collections import deque, Counter
import cv2
import joblib
import pyttsx3
from utils import make_detector, extract

CONF_THRESHOLD = 0.8   # predictions below this confidence are ignored
WINDOW = 10            # majority vote over the last 10 frames
MIN_VOTES = 7          # at least 7 matching frames needed
COOLDOWN = 2.0         # gap between two words (seconds)

clf = joblib.load("model.joblib")
detector = make_detector()
engine = pyttsx3.init()

window = deque(maxlen=WINDOW)
history = []
last_spoken, last_time = None, 0.0
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)

    feat, _ = extract(frame, detector)
    current = None
    if feat is not None:
        probs = clf.predict_proba([feat])[0]
        i = probs.argmax()
        if probs[i] >= CONF_THRESHOLD:
            current = clf.classes_[i]
    window.append(current)

    votes = Counter(w for w in window if w)
    if votes:
        word, count = votes.most_common(1)[0]
        if count >= MIN_VOTES and word != last_spoken and time.time() - last_time > COOLDOWN:
            history.append(word)
            engine.say(word)
            engine.runAndWait()
            last_spoken, last_time = word, time.time()
            window.clear()
    else:
        last_spoken = None   # hand removed, so the next sign can be spoken

    # On-screen output
    cv2.putText(frame, f"Sign: {current or '-'}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, "Text: " + " ".join(history[-6:]), (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Sign Language Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()