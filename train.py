import glob, os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay

X, y = [], []
for f in glob.glob("data/*.npy"):
    label = os.path.splitext(os.path.basename(f))[0]
    arr = np.load(f)
    X.extend(arr)
    y.extend([label] * len(arr))
X, y = np.array(X), np.array(y)
print("Classes:", sorted(set(y)), "| Total samples:", len(y))

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                      stratify=y, random_state=42)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(Xtr, ytr)

pred = clf.predict(Xte)
print("Test accuracy:", round(accuracy_score(yte, pred) * 100, 2), "%")
print(classification_report(yte, pred))

ConfusionMatrixDisplay.from_predictions(yte, pred, xticks_rotation=45)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
joblib.dump(clf, "model.joblib")
print("Model saved: model.joblib")