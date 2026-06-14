# Wake Word Detection

Dự án mẫu: phát hiện "wake word" (từ đánh thức) từ file âm thanh bằng trích xuất MFCC và các mô hình ML (SVM / RandomForest / KNN).

## Mô tả

Bộ script này cung cấp pipeline đơn giản để:

- Trích xuất đặc trưng MFCC từ các file âm thanh (1s)
- Huấn luyện mô hình phân loại (SVM/RandomForest/KNN) và lưu mô hình tốt nhất
- Phát hiện wake word trên file âm thanh bằng mô hình đã huấn luyện

## Yêu cầu

- Python 3.8+
- Thư viện: `numpy`, `librosa`, `scikit-learn`, `joblib`

Bạn có thể cài bằng pip:

```bash
pip install numpy librosa scikit-learn joblib
```

## Cấu trúc dự án

- `audio/` - chứa mẫu âm thanh chứa wake word (1s hoặc dài hơn)
- `noise/` - chứa file nền (sẽ được cắt thành đoạn 1s)
- `final_features/` - chứa file numpy đã trích xuất: `wake_word.npy`, `background.npy`
- `wakeword_preprocess.py` - script trích xuất MFCC và lưu feature
- `train_model.py` - script huấn luyện, tìm model tốt nhất và lưu `best_wakeword_model.pkl`
- `detect_from_file.py` - script dò wake word trên file (mặc định `test.wav`)

## Hướng dẫn nhanh

1. Chuẩn bị dữ liệu

- Thêm các file âm thanh chứa wake word vào `audio/` (định dạng `.wav` hoặc `.m4a`).
- Thêm file nền vào `noise/` (các file dài sẽ bị chia thành đoạn 1s).

2. Trích xuất đặc trưng

```bash
python wakeword_preprocess.py
```

Sau khi chạy sẽ tạo `final_features/wake_word.npy` và `final_features/background.npy`.

3. Huấn luyện mô hình

```bash
python train_model.py
```

Script sẽ chạy GridSearch trên một số mô hình, in kết quả và lưu mô hình tốt nhất thành `best_wakeword_model.pkl`.

4. Phát hiện wake word trên file

Chuẩn bị file `test.wav` (16 kHz) hoặc sửa `audio_path` trong `detect_from_file.py`.

```bash
python detect_from_file.py
```

Bạn có thể điều chỉnh các tham số trong `detect_from_file.py`:

- `window_size` (giây), `step_size` (giây)
- `threshold` (xác suất để coi là wake word)

## Ghi chú

- Pipeline dùng MFCC (13) + delta + delta2, và flatten thành vector cho mô hình ML scikit-learn.
- Ngưỡng phát hiện gợi ý: 0.75–0.9 (tùy dữ liệu). Script `train_model.py` in gợi ý threshold.
- Tăng cường dữ liệu (augmentation) đã được áp dụng trong `wakeword_preprocess.py` bằng cách thêm nhiễu Gaussian nhẹ.

## Liên hệ

Nếu cần hỗ trợ thêm hoặc muốn mở rộng sang mô hình deep learning, liên hệ tác giả hoặc tạo issue trên GitHub.

## License

Miễn trừ trách nhiệm — tùy chỉnh theo nhu cầu trước khi công khai.
