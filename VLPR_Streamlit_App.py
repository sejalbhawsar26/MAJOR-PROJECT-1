"""
VLPR Streamlit GUI - Single-file app
Requirements:
- Python 3.8+
- pip install streamlit opencv-python-headless ultralytics easyocr numpy pillow matplotlib
  (or opencv-python instead of headless if using locally)

How it works:
- Upload image or use webcam
- Detect license plates using YOLOv8 (ultralytics package). Uses 'yolov8n.pt' by default.
- Crop plate regions, preprocess, and run EasyOCR to read characters
- Shows detection boxes, cropped plates, and recognized text

Notes:
- For production or better accuracy, train a custom YOLO model on license plates and supply the weights path.
- You can swap EasyOCR for pytesseract if you prefer.

Run:
$ streamlit run VLPR_Streamlit_App.py

"""

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import easyocr
import tempfile
import os
import time

st.set_page_config(page_title="VLPR GUI", layout="wide")

# ---------- App config / constants ----------
DEFAULT_YOLO_WEIGHTS = 'yolov8n.pt'  # change to your custom weights like 'best.pt'
OCR_LANGS = ['en']
MIN_PLATE_AREA = 1000  # filter tiny boxes

# ---------- Helpers ----------
@st.cache_resource
def load_yolo(weights_path=DEFAULT_YOLO_WEIGHTS):
    try:
        model = YOLO(weights_path)
        return model
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        return None

@st.cache_resource
def load_ocr(langs=OCR_LANGS):
    reader = easyocr.Reader(langs, gpu=False)  # set gpu=True if available
    return reader

def np_from_pil(image: Image.Image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def pil_from_np(img_np):
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def draw_boxes(image_np, boxes, labels=None):
    img = image_np.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if labels is not None:
            text = labels[i]
            cv2.putText(img, text, (x1, max(10, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return img

# Preprocessing to improve OCR results
def preprocess_plate(plate_np):
    gray = cv2.cvtColor(plate_np, cv2.COLOR_BGR2GRAY)
    # Resize to reasonable size
    h, w = gray.shape
    scale = max(1, 200 / max(h, w))
    neww, newh = int(w*scale), int(h*scale)
    gray = cv2.resize(gray, (neww, newh), interpolation=cv2.INTER_LINEAR)
    # Denoise
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # Adaptive threshold
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY_INV, 11, 2)
    # Morphology to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Convert back to BGR for display consistency
    bgr = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    return bgr

# Run detection and OCR on an image (np BGR)
def run_vlpr(image_np, model, ocr_reader, conf_thres=0.25):
    results = model.predict(source=image_np, conf=conf_thres, imgsz=1280, device='cpu')
    # ultralytics returns a list of Results; we handle the first
    r = results[0]
    boxes = []
    scores = []
    # Ultralyics uses xyxy boxes
    if r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            boxes.append(xyxy.tolist())
            scores.append(conf)

    plate_images = []
    plate_texts = []
    plate_boxes_filtered = []

    h_img, w_img = image_np.shape[:2]

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(max(0, v)) for v in box]
        w = x2 - x1
        h = y2 - y1
        if w*h < MIN_PLATE_AREA:
            continue
        crop = image_np[y1:y2, x1:x2]
        # Preprocess and OCR
        prep = preprocess_plate(crop)
        # EasyOCR expects RGB
        prep_rgb = cv2.cvtColor(prep, cv2.COLOR_BGR2RGB)
        result = ocr_reader.readtext(prep_rgb)
        # Concatenate text results, prefer highest confidence words
        text_parts = []
        for (bbox, text, conf) in result:
            # Filter tiny fragments
            if len(text.strip()) >= 1:
                text_parts.append(text)
        text_final = ' '.join(text_parts).replace(' ', '')  # remove spaces between plate chars
        plate_images.append((crop, prep))
        plate_texts.append(text_final)
        plate_boxes_filtered.append((x1, y1, x2, y2))

    # Draw results
    drawn = draw_boxes(image_np, plate_boxes_filtered, labels=plate_texts)
    return drawn, plate_images, plate_texts, plate_boxes_filtered

# ---------- UI Layout ----------
st.title("🚘 Vehicle License Plate Recognition (VLPR) — GUI")

with st.sidebar:
    st.header("Settings")
    weights_path = st.text_input("YOLO weights path", value=DEFAULT_YOLO_WEIGHTS)
    conf = st.slider("Detection confidence threshold", 0.01, 1.0, 0.25)
    use_webcam = st.checkbox("Use webcam / camera input", value=False)
    show_crops = st.checkbox("Show cropped plate images", value=True)
    run_button = st.button("Run / Refresh")

# Load models (caches)
model = load_yolo(weights_path)
ocr = load_ocr()

col1, col2 = st.columns([1,1])

input_image = None

if use_webcam:
    cam_file = st.camera_input("Point your camera at a vehicle / plate")
    if cam_file:
        img = Image.open(cam_file)
        input_image = np_from_pil(img)
else:
    uploaded = st.file_uploader("Upload image (jpg/png)", type=["jpg","jpeg","png"] )
    if uploaded is not None:
        img = Image.open(uploaded).convert('RGB')
        input_image = np_from_pil(img)

# If run_button is pressed or input available, run
if input_image is not None and model is not None and ocr is not None and (run_button or True):
    t0 = time.time()
    with st.spinner('Running detection + OCR...'):
        drawn, plate_images, plate_texts, plate_boxes = run_vlpr(input_image, model, ocr, conf_thres=conf)
    t1 = time.time()
    st.success(f"Done — inference time: {t1-t0:.2f}s | Plates found: {len(plate_texts)}")

    # Show results
    col1.subheader("Result")
    col1.image(pil_from_np(drawn), use_column_width=True)

    col2.subheader("Recognized Plates")
    if len(plate_texts) == 0:
        col2.info("No plates detected. Try increasing the confidence threshold or upload a clearer image.")
    else:
        for i, txt in enumerate(plate_texts):
            st.markdown(f"**Plate {i+1}:** `{txt if txt!='' else '---'}`")
            if show_crops:
                crop, prep = plate_images[i]
                c1, c2 = st.columns(2)
                c1.image(pil_from_np(crop), caption="Cropped plate", use_column_width=True)
                c2.image(pil_from_np(prep), caption="Preprocessed for OCR", use_column_width=True)

    # Debug / detailed boxes
    with st.expander("Show plate bounding boxes (coords)"):
        for i, box in enumerate(plate_boxes):
            st.write(f"Plate {i+1} box: {box}")

else:
    st.info("Upload an image or enable webcam input and press Run. The app loads a YOLO model—first load may take a few seconds.")

st.write('---')
st.caption('Tip: For best results train a YOLO model specifically for your country/region license plates and supply the weights path above.')

# Footer: small help
st.markdown("**Dependencies**: `streamlit`, `ultralytics`, `opencv-python-headless`, `easyocr`, `numpy`, `Pillow`\n\nIf you want a version that uses pytesseract instead of EasyOCR or an example with video stream processing (MP4), ask and I will extend it.")
