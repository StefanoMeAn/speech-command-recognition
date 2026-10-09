"""
Single-command speech recognition demo using CNN + STFT.

Records one second of microphone audio, extracts STFT features,
and predicts one of the 35 speech commands.

Run:
    python demo.py
"""

from pathlib import Path
import sys

import numpy as np
import sounddevice as sd
import tensorflow as tf


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocessing import extract_stft, extract_mfcc

MODEL_PATH = PROJECT_ROOT / "models" / "cnn_stft.keras"

LABELS_PATH = PROJECT_ROOT / "models" / "labels.json"

SAMPLE_RATE = 16000
DURATION = 1.0


# ============================================================
# Load model
# ============================================================

import json

print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH) as f:
    labels = json.load(f)

print("Model loaded successfully!")
print(f"Number of commands: {len(labels)}")


# ============================================================
# Record audio
# ============================================================

def record_audio():

    print("\n🎤 Speak now!")

    recording = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )

    sd.wait()

    print("Recording complete.")

    return recording.flatten()


# ============================================================
# Preprocess audio
# ============================================================

def preprocess_audio(audio):

    features = extract_stft(audio)

    # Add batch dimension
    features = np.expand_dims(features, axis=0)

    return features


# ============================================================
# Predict command
# ============================================================

def predict_command(audio):

    features = preprocess_audio(audio)

    probabilities = model.predict(
        features,
        verbose=0,
    )[0]

    top_indices = np.argsort(probabilities)[::-1][:5]

    predicted_index = top_indices[0]

    predicted_command = labels[predicted_index]

    confidence = probabilities[predicted_index]

    print("\n" + "=" * 45)

    print(f"Predicted command: {predicted_command.upper()}")

    print(f"Confidence: {confidence:.2%}")

    print("\nTop 5 predictions:")

    for rank, index in enumerate(top_indices, start=1):

        print(
            f"{rank}. {labels[index]:15s}"
            f"{probabilities[index]:.2%}"
        )

    print("=" * 45)

    return predicted_command


# ============================================================
# Interactive application
# ============================================================

def main():

    print("\nSpeech Command Recognition Demo")
    print("--------------------------------")

    print("Press ENTER to record a command.")
    print("Type 'q' to quit.")

    while True:

        user_input = input("\nPress ENTER to speak: ")

        if user_input.lower() == "q":
            print("Goodbye!")
            break

        audio = record_audio()

        predict_command(audio)


if __name__ == "__main__":
    main()