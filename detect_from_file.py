import librosa
import numpy as np
import joblib

# load model
model = joblib.load("best_wakeword_model.pkl")

# load audio
audio_path = "test.wav"
y, sr = librosa.load(audio_path, sr=16000)

print("Audio length:", len(y)/sr, "seconds")

# sliding window
window_size = 1.0
step_size = 0.1
threshold = 0.8

window_samples = int(window_size * sr)
step_samples = int(step_size * sr)

detections = []

for start in range(0, len(y) - window_samples, step_samples):

    segment = y[start:start + window_samples]

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=segment,
        sr=sr,
        n_mfcc=13
    )

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.vstack([mfcc, delta, delta2])

    features = features.flatten()

    features = features.reshape(1, -1)

    prob = model.predict_proba(features)[0][1]

    if prob > 0.65:
        time_sec = start / sr
        detections.append(time_sec)

# lọc detection gần nhau
filtered = []
min_gap = 1.5
for t in detections:
    if not filtered or t - filtered[-1] > min_gap:
        filtered.append(t)

# print result
if filtered:
    print("\nWake word detected at:")
    for t in filtered:
        print(f"{t:.2f} seconds")
else:
    print("\nNo wake word detected")