"""
OPTIONAL: Train your own 4-class emotion CNN from scratch on FER-2013.

This is NOT required to run main.py — a working pretrained model
(emotion_model.h5) is already included in this project. Use this script
only if you want to demonstrate the *training* side of the pipeline for
your seminar, or if you want a model that natively outputs exactly
4 classes at 48x48 (instead of the bundled 7-class/64x64 model that
main.py filters down to 4 classes at inference time).

Steps to use:
    1. Download fer2013.csv from Kaggle:
       https://www.kaggle.com/datasets/msambare/fer2013
       (or the original "Challenges in Representation Learning: Facial
       Expression Recognition Challenge" dataset)
    2. Place fer2013.csv in this project folder.
    3. Run:  python train_model.py
    4. It will save a new "emotion_model.h5" (48x48 input, 4-class output)
       that main.py can load automatically (input size is auto-detected).

Training takes roughly 15-30 minutes on a laptop CPU for ~25 epochs,
much faster with a GPU. Feel free to lower EPOCHS for a quicker demo.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

CSV_PATH = "fer2013.csv"
OUTPUT_MODEL_PATH = "emotion_model.h5"
IMG_SIZE = 48
EPOCHS = 25
BATCH_SIZE = 64

# FER-2013 original label indices -> our 4 target classes
# Original FER2013 labels: 0=Angry 1=Disgust 2=Fear 3=Happy 4=Sad 5=Surprise 6=Neutral
FER_TO_TARGET = {
    0: 0,  # Angry   -> Angry (0)
    3: 1,  # Happy   -> Happy (1)
    4: 2,  # Sad     -> Sad (2)
    6: 3,  # Neutral -> Neutral (3)
}
TARGET_NAMES = ["Angry", "Happy", "Sad", "Neutral"]


def load_fer2013(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"'{csv_path}' not found. Download it from Kaggle "
            "(search 'fer2013') and place it in this folder."
        )
    df = pd.read_csv(csv_path)
    df = df[df["emotion"].isin(FER_TO_TARGET.keys())].reset_index(drop=True)

    pixels = df["pixels"].apply(lambda s: np.array(s.split(), dtype="float32"))
    X = np.stack(pixels.values)
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0

    y_raw = df["emotion"].map(FER_TO_TARGET).values
    y = tf.keras.utils.to_categorical(y_raw, num_classes=len(TARGET_NAMES))

    return X, y


def build_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(len(TARGET_NAMES), activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("Loading FER-2013 dataset (Angry / Happy / Sad / Neutral only)...")
    X, y = load_fer2013(CSV_PATH)
    print(f"Loaded {len(X)} labeled face images.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    model = build_model()
    model.summary()

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(X_train)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(
            OUTPUT_MODEL_PATH, save_best_only=True, monitor="val_accuracy"
        ),
    ]

    model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    model.save(OUTPUT_MODEL_PATH)
    print(f"\nSaved trained model to '{OUTPUT_MODEL_PATH}'.")
    print("main.py will automatically detect the new 48x48 / 4-class input shape.")


if __name__ == "__main__":
    main()
