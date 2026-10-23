# Real-Time Sign Language Translator for Classrooms

## Problem
Hearing/speech-impaired students find it hard to ask doubts in class.

## Solution
We track hand gestures from a webcam (MediaPipe, 21 landmarks), predict the sign
with an ML classifier, and convert it into text + speech.

## Scope
- Isolated signs (~10 words): hello, doubt, yes, no, repeat, help, ...
- Sign language: ISL (or whichever you chose)

## Results
- Dataset: X samples, Y participants
- Test accuracy: __%
- Average latency: __ ms | FPS: __

## Limitations
- Effect of lighting and background
- Isolated signs only, no continuous signing
- Single camera angle

## Future Scope
Continuous recognition with LSTM/Transformer, mobile app, larger vocabulary.

## Team
| Member | Role |
|---|---|
| Name 1 | Data collection, dataset |
| Name 2 | Model training, evaluation |
| Name 3 | App + TTS integration |
| Name 4 | Report, PPT, testing |
