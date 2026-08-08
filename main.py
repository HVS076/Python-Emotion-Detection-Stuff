"""
Real-Time Facial Emotion Detection AI
--------------------------------------
A beginner-friendly desktop app that:
  1. Captures live video from a webcam (OpenCV)
  2. Detects faces in each frame (Haar Cascade)
  3. Crops + preprocesses each face (grayscale, 48x48 or 64x64 depending on model)
  4. Classifies the emotion using a pretrained CNN (TensorFlow/Keras)
  5. Displays everything live in a Tkinter GUI

Emotions shown: Happy, Sad, Angry, Neutral

Author: (your name here)
For: AI/ML Seminar Demo
"""

import os
import sys
import time
import collections

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

# TensorFlow can be a bit noisy on import - silence info logs before importing
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.h5")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface.xml")

# The 4 emotions this project reports, in the order the seminar spec asked for.
TARGET_EMOTIONS = ["Angry", "Happy", "Sad", "Neutral"]

# Colors (BGR, for OpenCV drawing) per emotion, so the rectangle changes color
EMOTION_COLORS_BGR = {
    "Angry": (0, 0, 255),      # red
    "Happy": (0, 200, 0),      # green
    "Sad": (255, 140, 0),      # blue-ish
    "Neutral": (200, 200, 200),  # gray
}

CAM_INDEX = 0            # change to 1, 2... if you have multiple cameras
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DETECT_EVERY_N_FRAMES = 1  # set to 2-3 on slower laptops to boost FPS


# --------------------------------------------------------------------------
# Emotion Model Wrapper
# --------------------------------------------------------------------------
class EmotionModel:
    """
    Loads the pretrained CNN and exposes a single `.predict(face_gray)` call
    that returns (emotion_label, confidence_float, full_prob_dict).

    The bundled emotion_model.h5 is the open-source "mini_XCEPTION" FER2013
    model (MIT licensed, oarriaga/face_classification project). It was
    trained on 7 emotions (angry, disgust, fear, happy, sad, surprise,
    neutral) at 64x64 grayscale input. For this seminar project we only
    care about 4 of those classes, so we:
        1. Run the full 7-class prediction
        2. Keep only Angry / Happy / Sad / Neutral
        3. Re-normalize those 4 probabilities so they sum to 100%

    If you swap in your own model trained from scratch on FER2013 with a
    48x48 grayscale input and a 4-unit softmax output, this class will
    auto-detect the new input size and skip the 7->4 filtering step.
    """

    # Class order used by the bundled mini_XCEPTION model
    FULL_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Could not find model file at '{model_path}'.\n"
                "Place your trained/downloaded emotion_model.h5 in the project folder."
            )

        self.model = tf.keras.models.load_model(model_path, compile=False)

        # Figure out what input size/shape this model expects, e.g. (None, 64, 64, 1)
        input_shape = self.model.input_shape
        self.input_h = input_shape[1]
        self.input_w = input_shape[2]
        self.output_units = self.model.output_shape[-1]

        self.is_full_7_class = self.output_units == 7

        print(f"[EmotionModel] Loaded model. Expected input: {self.input_h}x{self.input_w}, "
              f"output classes: {self.output_units}")

    def preprocess(self, face_gray: np.ndarray) -> np.ndarray:
        """Resize + normalize a grayscale face crop for the model."""
        face = cv2.resize(face_gray, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA)
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=-1)  # (H, W, 1)
        face = np.expand_dims(face, axis=0)   # (1, H, W, 1)
        return face

    def predict(self, face_gray: np.ndarray):
        x = self.preprocess(face_gray)
        preds = self.model.predict(x, verbose=0)[0]  # shape (num_classes,)

        if self.is_full_7_class:
            full_probs = dict(zip(self.FULL_LABELS, preds))
            # Keep only the 4 target emotions and re-normalize
            filtered = {k: full_probs[k] for k in TARGET_EMOTIONS}
            total = sum(filtered.values()) + 1e-8
            probs = {k: (v / total) for k, v in filtered.items()}
        else:
            # Assume the model was trained/exported with exactly the 4 target classes
            probs = dict(zip(TARGET_EMOTIONS, preds))
            total = sum(probs.values()) + 1e-8
            probs = {k: v / total for k, v in probs.items()}

        best_label = max(probs, key=probs.get)
        best_conf = probs[best_label] * 100.0
        return best_label, best_conf, probs


# --------------------------------------------------------------------------
# Main Application (Tkinter GUI)
# --------------------------------------------------------------------------
class EmotionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Real-Time Facial Emotion Detection AI")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(False, False)

        # ---- Load AI components (with graceful error handling) ----
        try:
            self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
            if self.face_cascade.empty():
                raise IOError(f"Failed to load Haar Cascade from {CASCADE_PATH}")
        except Exception as e:
            messagebox.showerror("Startup Error", f"Could not load face detector:\n{e}")
            root.destroy()
            sys.exit(1)

        try:
            self.emotion_model = EmotionModel(MODEL_PATH)
        except Exception as e:
            messagebox.showerror("Startup Error", f"Could not load emotion model:\n{e}")
            root.destroy()
            sys.exit(1)

        try:
            self.cap = cv2.VideoCapture(CAM_INDEX)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            if not self.cap.isOpened():
                raise IOError("Webcam could not be opened. Is it in use by another app?")
        except Exception as e:
            messagebox.showerror("Webcam Error", str(e))
            root.destroy()
            sys.exit(1)

        # Small rolling history for a simple emotion trend readout
        self.history = collections.deque(maxlen=30)
        self.frame_count = 0
        self.last_result = ("Neutral", 0.0, {e: 0.0 for e in TARGET_EMOTIONS})

        self._build_gui()
        self._update_loop()

    # ---------------- GUI layout ----------------
    def _build_gui(self):
        title = tk.Label(
            self.root, text="🎭 Real-Time Emotion Detector AI",
            font=("Segoe UI", 18, "bold"), fg="white", bg="#1e1e2e", pady=10
        )
        title.grid(row=0, column=0, columnspan=2, sticky="ew")

        # Left: video feed
        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.grid(row=1, column=0, padx=15, pady=10)

        # Right: info panel
        info_frame = tk.Frame(self.root, bg="#282a3a", width=260)
        info_frame.grid(row=1, column=1, padx=(0, 15), pady=10, sticky="n")

        tk.Label(info_frame, text="Emotion:", font=("Segoe UI", 12),
                 fg="#aaaaaa", bg="#282a3a").pack(anchor="w", padx=15, pady=(15, 0))
        self.emotion_var = tk.StringVar(value="—")
        tk.Label(info_frame, textvariable=self.emotion_var, font=("Segoe UI", 26, "bold"),
                 fg="#f5c542", bg="#282a3a").pack(anchor="w", padx=15)

        tk.Label(info_frame, text="Confidence:", font=("Segoe UI", 12),
                 fg="#aaaaaa", bg="#282a3a").pack(anchor="w", padx=15, pady=(10, 0))
        self.conf_var = tk.StringVar(value="—")
        tk.Label(info_frame, textvariable=self.conf_var, font=("Segoe UI", 20, "bold"),
                 fg="white", bg="#282a3a").pack(anchor="w", padx=15)

        ttk.Separator(info_frame, orient="horizontal").pack(fill="x", padx=15, pady=15)

        # Confidence bars for every emotion (nice for the seminar demo)
        tk.Label(info_frame, text="All Emotion Scores", font=("Segoe UI", 11, "bold"),
                 fg="#aaaaaa", bg="#282a3a").pack(anchor="w", padx=15)

        self.bar_vars = {}
        self.bar_widgets = {}
        for emo in TARGET_EMOTIONS:
            row = tk.Frame(info_frame, bg="#282a3a")
            row.pack(fill="x", padx=15, pady=4)
            tk.Label(row, text=emo, width=8, anchor="w", font=("Segoe UI", 10),
                     fg="white", bg="#282a3a").pack(side="left")
            pb = ttk.Progressbar(row, length=120, maximum=100)
            pb.pack(side="left", padx=5)
            val_lbl = tk.Label(row, text="0%", width=5, font=("Segoe UI", 9),
                                fg="#aaaaaa", bg="#282a3a")
            val_lbl.pack(side="left")
            self.bar_widgets[emo] = pb
            self.bar_vars[emo] = val_lbl

        ttk.Separator(info_frame, orient="horizontal").pack(fill="x", padx=15, pady=15)

        tk.Label(info_frame, text="Model:", font=("Segoe UI", 10),
                 fg="#aaaaaa", bg="#282a3a").pack(anchor="w", padx=15)
        tk.Label(info_frame, text="CNN + OpenCV Haar Cascade", font=("Segoe UI", 10, "bold"),
                 fg="white", bg="#282a3a").pack(anchor="w", padx=15, pady=(0, 15))

        self.status_var = tk.StringVar(value="Starting camera...")
        status_bar = tk.Label(self.root, textvariable=self.status_var, font=("Segoe UI", 9),
                               fg="#777777", bg="#1e1e2e", anchor="w")
        status_bar.grid(row=2, column=0, columnspan=2, sticky="ew", padx=15, pady=(0, 10))

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------- Main video/inference loop ----------------
    def _update_loop(self):
        start = time.time()
        ret, frame = self.cap.read()

        if not ret:
            self.status_var.set("⚠ Failed to read from webcam.")
            self.root.after(500, self._update_loop)
            return

        frame = cv2.flip(frame, 1)  # mirror, feels more natural
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        if len(faces) == 0:
            self.status_var.set("No face detected — center your face in frame.")
        else:
            # Use the largest detected face (closest to camera)
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            for i, (x, y, w, h) in enumerate(faces):
                face_roi = gray[y:y + h, x:x + w]
                if face_roi.size == 0:
                    continue

                try:
                    label, conf, probs = self.emotion_model.predict(face_roi)
                except Exception as e:
                    self.status_var.set(f"Prediction error: {e}")
                    continue

                color = EMOTION_COLORS_BGR.get(label, (255, 255, 255))
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {conf:.1f}%", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if i == 0:  # only update the side panel with the primary face
                    self.last_result = (label, conf, probs)
                    self.history.append(label)

            self.status_var.set(f"{len(faces)} face(s) detected.")

        self._render_frame(frame)
        self._update_side_panel(*self.last_result)

        elapsed = time.time() - start
        delay_ms = max(1, int((1 / 30 - elapsed) * 1000))  # aim for ~30 FPS
        self.root.after(delay_ms, self._update_loop)

    def _render_frame(self, bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk  # keep a reference! otherwise it gets garbage collected
        self.video_label.configure(image=imgtk)

    def _update_side_panel(self, label, conf, probs):
        self.emotion_var.set(label)
        self.conf_var.set(f"{conf:.1f}%")
        for emo, pb in self.bar_widgets.items():
            val = probs.get(emo, 0.0) * 100.0
            pb["value"] = val
            self.bar_vars[emo]["text"] = f"{val:.0f}%"

    def _on_close(self):
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main():
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
