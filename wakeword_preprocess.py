import os
import librosa
import numpy as np

# ===== CẤU HÌNH =====
INPUT_FOLDERS = {
    "wake_word": "audio",
    "background": "noise"
}

OUTPUT_DIR = "final_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SR = 16000
DURATION = 1.0
TARGET_SAMPLES = int(TARGET_SR * DURATION)

N_MFCC = 13

def extract_features(y):

    y = librosa.util.normalize(y)

    mfcc = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.vstack([mfcc, delta, delta2])

    return features.flatten()


# ===== XỬ LÝ =====
for label, folder in INPUT_FOLDERS.items():

    if not os.path.exists(folder):
        continue

    all_features = []

    files = [f for f in os.listdir(folder)
             if f.lower().endswith(('.wav', '.m4a'))]

    for f in files:

        path = os.path.join(folder, f)

        y_raw, sr = librosa.load(path, sr=TARGET_SR)

        if label == "background":

            num_segments = len(y_raw) // TARGET_SAMPLES

            print(f"✂️ Đang chia {f} thành {num_segments} đoạn noise")

            for i in range(num_segments):

                segment = y_raw[i*TARGET_SAMPLES:(i+1)*TARGET_SAMPLES]

                all_features.append(extract_features(segment))

        else:

            y_trimmed, _ = librosa.effects.trim(y_raw, top_db=20)

            if len(y_trimmed) < TARGET_SAMPLES:
                y_trimmed = np.pad(y_trimmed,
                                   (0, TARGET_SAMPLES - len(y_trimmed)))

            y_final = y_trimmed[:TARGET_SAMPLES]

            all_features.append(extract_features(y_final))

            # DATA AUGMENTATION (rất quan trọng)
            noise = np.random.normal(0, 0.005, y_final.shape)
            augmented = y_final + noise

            all_features.append(extract_features(augmented))

    if all_features:

        np.save(os.path.join(OUTPUT_DIR, f"{label}.npy"),
                np.array(all_features))

        print(f"✅ Lưu {len(all_features)} samples -> {label}.npy")


print("\n🚀 Feature extraction hoàn tất!")