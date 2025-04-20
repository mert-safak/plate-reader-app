import streamlit as st
from PIL import Image
from helper import detect_plate, read_plate_text

st.set_page_config(page_title="Plaka Okuma Sistemi", layout="centered")
st.title("🚗 Plaka Tanıma ve Okuma Sistemi")

file = st.file_uploader("Bir araç görüntüsü yükleyin", type=["jpg", "jpeg", "png"])

# Model yolları (Hugging Face'ten indirme için doğru path)
plate_model_path = "Models/plate_detection.pt"
char_model_path = "Models/plate_reading.pt"

if file is not None:
    st.header("Yüklenen Görsel")
    image = Image.open(file).convert("RGB")
    st.image(image, use_container_width=True)

    st.header("Plaka Tespiti")
    detection_result, cropped_image, is_detected = detect_plate(image, plate_model_path)
    st.image(detection_result, caption="Tespit Edilen Plaka", use_container_width=True)

    if is_detected > 0 and cropped_image is not None:
        st.subheader("Kırpılmış Plaka")
        st.image(cropped_image, use_container_width=True)

        st.subheader("Okunan Plaka Yazısı")
        plate_text = read_plate_text(cropped_image, char_model_path)
        st.success(f"🅿️ Plaka: {plate_text}")
    else:
        st.warning("Plaka bulunamadı.")
