# 🎭 Real-Time Facial Emotion Detection AI

A beginner-friendly desktop application that uses your webcam to detect a
face and classify its emotion — **Happy, Sad, Angry, or Neutral** — in real
time, with a live Tkinter GUI showing the video feed, the predicted
emotion, and confidence bars for every class.

Built for an AI/ML seminar demo: simple enough to explain in 10 minutes,
visual enough to impress an audience.

```
Emotion_AI/
│
├── main.py                     # Run this — the full application
├── train_model.py              # OPTIONAL — train your own model on FER-2013
├── emotion_model.h5             # Pretrained CNN (already included)
├── haarcascade_frontalface.xml  # OpenCV face detector (already included)
├── requirements.txt
└── README.md
```

---

## 1. How it works (architecture)

```
Webcam
  │
  ▼
OpenCV captures frame
  │
  ▼
Haar Cascade face detection  → finds (x, y, w, h) of each face
  │
  ▼
Crop face → convert to grayscale → resize to model's input size
  │
  ▼
CNN (Keras) predicts emotion probabilities
  │
  ▼
Pick highest-probability emotion + confidence %
  │
  ▼
Tkinter GUI updates: video frame, colored box, emotion label, confidence bars
```

This whole loop runs roughly 30 times per second, so it feels live.

---

## 2. About the pretrained model

`emotion_model.h5` included in this project is the open-source
**mini_XCEPTION** CNN trained on the FER-2013 dataset
(from the MIT-licensed `oarriaga/face_classification` project). It was
originally trained to recognize **7** emotions (angry, disgust, fear,
happy, sad, surprise, neutral) on **64×64** grayscale faces.

Since this seminar project only needs **4** emotions, `main.py` runs the
full 7-class prediction internally and then:
1. Keeps only Angry / Happy / Sad / Neutral
2. Re-normalizes those 4 scores so they add up to 100%

This means you get a genuinely working, reasonably accurate real-time demo
**without needing to train anything yourself**.

**Want to train your own model instead?** Run `train_model.py` — it
trains a small CNN from scratch on the raw FER-2013 CSV, using exactly
the 48×48 grayscale input / 4-class softmax output shape described in the
assignment. `main.py` auto-detects whichever model you drop in (it reads
`model.input_shape` at load time), so you can swap models with zero code
changes. See the comments at the top of `train_model.py` for the dataset
download link and instructions.

---

## 3. Installation

### Step 1 — Install Python
Python 3.9–3.11 recommended (TensorFlow does not yet fully support every
new Python release day-one).

### Step 2 — Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Check the required files are in place
Both files are already included in this folder — you don't need to
download anything extra:
- `emotion_model.h5` — the pretrained CNN
- `haarcascade_frontalface.xml` — the OpenCV face detector

(If you ever need to re-download the cascade file yourself, it also
ships with `opencv-python` — you can find it via
`cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'`.)

### Step 5 — Run it
```bash
python main.py
```

A window will open showing your webcam feed. Center your face in frame —
a colored rectangle will appear around it with the detected emotion and
confidence percentage. The side panel shows a live breakdown across all
4 emotions.

Press the window's close button (X) to quit — this properly releases the
webcam.

---

## 4. Troubleshooting

| Problem | Fix |
|---|---|
| `Webcam could not be opened` | Close other apps using the camera (Zoom, Teams, etc.), or change `CAM_INDEX` in `main.py` to `1`. |
| Window opens but is black / frozen | Some laptops need a moment for the camera driver to initialize — wait a few seconds; if it persists, try a different `CAM_INDEX`. |
| `ModuleNotFoundError` | Make sure your virtual environment is activated, then re-run `pip install -r requirements.txt`. |
| Low FPS / laggy | Increase `DETECT_EVERY_N_FRAMES` in `main.py`, or shrink `FRAME_WIDTH`/`FRAME_HEIGHT`. |
| Poor lighting reduces accuracy | Face the camera toward a light source, not away from it. |

---

## 5. Seminar talking points

**What is a CNN (Convolutional Neural Network)?**
A CNN is a type of deep learning model built to understand images. Instead
of looking at every pixel independently, it slides small filters
("kernels") across the image to detect patterns — edges first, then
shapes like eyes or mouth curves, then whole facial expressions in deeper
layers. Pooling layers shrink the image progressively so the network
learns increasingly abstract features, and the final dense layers turn
those features into a probability for each emotion class.

**How does facial emotion recognition work here?**
1. A face is detected and isolated from the background.
2. The face is converted to grayscale and resized to a fixed size (color
   and exact resolution aren't essential for expression — shape and
   intensity patterns are).
3. The CNN, trained on thousands of labeled FER-2013 face images, outputs
   a probability for each emotion class.
4. The highest probability becomes the predicted label; its value becomes
   the "confidence."

**How is OpenCV used?**
- `cv2.VideoCapture` grabs frames from the webcam.
- `cv2.CascadeClassifier` (Haar Cascade) is a fast, classical
  computer-vision algorithm — not deep learning — that scans the image at
  multiple scales for face-like patterns of light/dark regions. It's much
  lighter than a deep face detector, which makes it perfect for real-time
  use.
- `cv2.rectangle` / `cv2.putText` draw the bounding box and label directly
  onto each frame before it's shown in the GUI.

**Why FER-2013?**
FER-2013 is a well-known public dataset of ~35,000 grayscale 48×48 facial
images labeled with 7 emotions, originally released for a Kaggle
competition. It's a standard benchmark for teaching facial emotion
recognition because it's free, moderately sized, and diverse.

**Real-world applications**
- Customer sentiment analysis in retail or call centers
- Driver drowsiness/frustration monitoring in cars
- Mental health / wellbeing check-in tools
- Adaptive learning platforms that respond to student engagement
- Human-computer interaction — smarter, emotion-aware assistants
- Market research (measuring reactions to ads or products)

---

## 6. Optional enhancements (ideas + where to start)

These aren't required, but they're great "bonus" additions to show off
extra effort in a seminar:

- **Emotion history graph** — keep a rolling `collections.deque` of recent
  predictions (already scaffolded as `self.history` in `main.py`) and plot
  it with `matplotlib` embedded in a Tkinter `Canvas`, or a simple bar of
  emotion counts over the last N seconds.
- **Multiple face detection** — `main.py` already detects *all* faces each
  frame (`faces = self.face_cascade.detectMultiScale(...)`) and draws a
  box on each one; extend the side panel to list every face's emotion
  instead of just the largest one.
- **Voice feedback** — use `pyttsx3` to speak the detected emotion out
  loud when it changes (`engine.say(f"You look {label}")`), throttled so
  it doesn't repeat every frame.
- **Modern GUI** — swap `tkinter`/`ttk` widgets for `customtkinter` for
  rounded corners, dark mode, and nicer fonts with almost no code changes.
- **Confidence bars** — already implemented! See the `ttk.Progressbar`
  widgets in the side panel.
- **Save detection history** — append `(timestamp, emotion, confidence)`
  rows to a CSV file each frame (or once per second) using Python's
  built-in `csv` module, so you can show a "session summary" at the end
  of the demo.

---

## 7. Credits

- Face detector: OpenCV's `haarcascade_frontalface_default.xml`
- Pretrained CNN: mini_XCEPTION, from the MIT-licensed
  `oarriaga/face_classification` project, trained on FER-2013
- Built with: Python, TensorFlow/Keras, OpenCV, Tkinter, Pillow, NumPy
