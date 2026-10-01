import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="Detecção de Objetos - PoC Marketing",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

def process_image(image_bytes, model, conf_threshold):
    img = Image.open(image_bytes).convert("RGB")
    img_np = np.array(img)
    
    results = model.predict(source=img_np, conf=conf_threshold, verbose=False)
    
    res = results[0]
    res_bgr = res.plot()
    res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
    
    classes_detected = {}
    if res.boxes is not None:
        for box in res.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            classes_detected[cls_name] = classes_detected.get(cls_name, 0) + 1
            
    return res_rgb, classes_detected

st.title("🔍 Sistema de Detecção de Objetos em Imagens")
st.markdown("Protótipo leve para identificação de conteúdo visual em campanhas publicitárias.")

model = load_model()

st.sidebar.header("Configurações")
conf_thresh = st.sidebar.slider("Limiar de Confiança (Confidence)", 0.1, 1.0, 0.25, 0.05)

uploaded_file = st.file_uploader("Envie uma imagem (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagem Original")
        st.image(uploaded_file, use_container_width=True)
        
    with st.spinner("Processando imagem..."):
        annotated_img, detections = process_image(uploaded_file, model, conf_thresh)
        
    with col2:
        st.subheader("Objetos Detectados")
        st.image(annotated_img, use_container_width=True)
        
    st.markdown("---")
    st.subheader("📊 Resumo das Detecções")
    
    if detections:
        cols = st.columns(min(len(detections), 4))
        for idx, (obj_name, count) in enumerate(detections.items()):
            with cols[idx % 4]:
                st.metric(label=obj_name.capitalize(), value=count)
    else:
        st.info("Nenhum objeto detectado com o limiar de confiança atual.")