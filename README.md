# Real-Time AI Sign Language Translator for Classrooms

**Institution:** Gautam Buddha University
**Degree:** B.Tech CSE (Specialization in AI)

## Problem
Lakhs of hearing and speech-impaired students face communication barriers in
classrooms. Asking a doubt often needs a teacher or interpreter, which is not
always available.

## Solution
A webcam tracks the student's hand gestures in real time. MediaPipe extracts
21 hand landmarks, a trained classifier predicts the sign, and the result is
shown as text on screen and spoken aloud through text-to-speech.

## Pipeline
Video Capture → Hand Landmark Detection (MediaPipe) → AI Classifier (scikit-learn) → Text + Speech

## Two-Phase Approach
1. **Phase 1 (Quick Win):** Google Teachable Machine prototype as a backup demo.
2. **Phase 2 (Upgrade):** Python + OpenCV + MediaPipe for landmark-based recognition.

## Scope
- Isolated signs only (about 10 words): hello, doubt, yes, no, ...
- Sign language: ISL (or whichever was chosen)

## Results
- Dataset: 16 samples, 4 participants
- Test accuracy: modrate
- Average latency: 21.3 ms | FPS: 46.9

## Limitations
- Lighting and background affect detection accuracy
- Isolated signs only, no continuous signing
- Single camera angle
- Model is trained only on the signers in the dataset

## Future Scope
- Continuous sign recognition with LSTM or Transformer models
- Mobile app
- Larger vocabulary

## Team
| Name | Role |
|---|---|
| Hemansh | Core ML & Deployment Lead |
| Durga | Video Processing & Pipeline Engineer |
| Divay Singh Bhati | Biometric Landmark Engineer |
| Anshi Goyal | UX Logic & System Integration Engineer |

## How to Run
```bash
py -3.11 -m venv venv
venv\Scripts\python.exe app.py
```
