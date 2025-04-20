import os
import urllib.request
import cv2
import numpy as np
from ultralytics import YOLO

# Model indirme fonksiyonu
def download_model(url, output_path):
    if not os.path.exists(output_path):
        print(f"[INFO] Model indiriliyor: {url}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        urllib.request.urlretrieve(url, output_path)
        print(f"[INFO] Model indirildi: {output_path}")

# Hugging Face'ten modelleri indir
plate_model_url = "https://huggingface.co/mertsafak/plate-reader-model/resolve/main/plate_detection.pt"
char_model_url = "https://huggingface.co/mertsafak/plate-reader-model/resolve/main/plate_reading.pt"
plate_model_path = "Models/plate_detection.pt"
char_model_path = "Models/plate_reading.pt"

download_model(plate_model_url, plate_model_path)
download_model(char_model_url, char_model_path)

# Modelleri yükle
plate_model = YOLO(plate_model_path)
char_model = YOLO(char_model_path)

# Plaka tespiti
def detect_plate(image):
    green = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    image_original = np.asarray(image).copy()
    image_array = image_original.copy()
    results = plate_model(image_array)[0]

    cropped_image = None
    is_detected = 0

    boxes = results.boxes.data.tolist()
    if boxes:
        for result in boxes:
            x1, y1, x2, y2, score, class_id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            if score > 0.5:
                is_detected += 1
                pad = 5
                h, w = image_original.shape[:2]
                x1_pad = max(0, x1 - pad)
                y1_pad = max(0, y1 - pad)
                x2_pad = min(w, x2 + pad)
                y2_pad = min(h, y2 + pad)

                cropped_image = image_original[y1_pad:y2_pad, x1_pad:x2_pad].copy()
                cv2.rectangle(image_array, (x1, y1), (x2, y2), green, 2)
                text = f"{results.names[int(class_id)]}: %{score*100:.2f}"
                cv2.putText(image_array, text, (x1, y1 - 10), font, 0.5, green, 1, cv2.LINE_AA)

    else:
        cropped_image = np.zeros((512, 512, 3), np.uint8)
        cv2.putText(image_array, "No Detection", (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    return image_array, cropped_image, is_detected

# Karakter okuma
def read_plate_text(cropped_image):
    h, w = cropped_image.shape[:2]
    if h < 100 or w < 100:
        cropped_image = cv2.resize(cropped_image, (640, 640), interpolation=cv2.INTER_LINEAR)

    results = char_model.predict(source=cropped_image, imgsz=640, conf=0.25, verbose=False)[0]

    if results.boxes is None or len(results.boxes.cls) == 0:
        print("[INFO] Karakter bulunamadı.")
        return "[Okunamadı]"

    chars = results.boxes.cls
    boxes = results.boxes.xyxy
    ordered = sorted(zip(chars, boxes), key=lambda x: (x[1][0] + x[1][2]) / 2)

    text = ''.join([char_model.names[int(cls)] for cls, _ in ordered])
    print(f"[INFO] Okunan Plaka: {text}")

    for cls, box in ordered:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(cropped_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.putText(cropped_image, char_model.names[int(cls)], (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return text
