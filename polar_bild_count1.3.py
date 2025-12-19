import streamlit as st
import cv2
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from PIL import Image

st.set_page_config(page_title="PSR Collagen Quantification – Optimiert", layout="wide")
st.title("📊 PSR Polarized Collagen Quantifier – Optimierte Version")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Threshold-Modus")
mode = st.sidebar.radio("Modus", ["Auto+Offset", "Manuell"])
offset = st.sidebar.slider("Otsu Offset", -40, 40, -10)
manual_thresh = st.sidebar.slider("Manueller Threshold (V)", 0, 255, 110)

st.sidebar.header("Sättigungsfilter")
sat_min = st.sidebar.slider("Min. Saturation (S)", 0, 20, 5)

st.sidebar.header("Hue-Bereiche (OpenCV HSV)")
red_max = st.sidebar.slider("Rot max", 5, 15, 10)
orange_low = st.sidebar.slider("Orange low", 8, 20, 12)
orange_high = st.sidebar.slider("Orange high", 20, 40, 30)
green_low = st.sidebar.slider("Grün low", 30, 60, 40)
green_high = st.sidebar.slider("Grün high", 60, 120, 90)

st.sidebar.header("Objektfilter")
min_length = st.sidebar.slider("Minimale Faserlänge (px)", 1, 100, 10)
min_area = st.sidebar.slider("Minimale Fläche (px²)", 1, 20, 5)

uploaded = st.sidebar.file_uploader(
    "PSR Bilder hochladen",
    type=["tif", "tiff", "png", "jpg"],
    accept_multiple_files=True
)

# -------------------------
# Analyse-Funktion
# -------------------------
def analyze_image(file):
    # --- Safe image loading (TIFF compatible) ---
    img = Image.open(file)
    img = np.array(img.convert("RGB"), dtype=np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    v_uint8 = v.astype(np.uint8)

    # -------------------------
    # Thresholding
    # -------------------------
    if mode == "Manuell":
        mask_thresh = (v_uint8 > manual_thresh).astype(np.uint8) * 255
    else:
        otsu_val, _ = cv2.threshold(v_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_val = np.clip(otsu_val + offset, 0, 255)
        mask_thresh = (v_uint8 > thresh_val).astype(np.uint8) * 255

    adaptive_thresh = cv2.adaptiveThreshold(
        v_uint8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2
    )

    combined_mask = cv2.bitwise_or(mask_thresh, adaptive_thresh)

    # Sättigungsfilter
    sat_mask = (s > sat_min).astype(np.uint8) * 255
    collagen_mask = cv2.bitwise_and(combined_mask, sat_mask)

    # Morphologische Reinigung
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    collagen_mask = cv2.morphologyEx(collagen_mask, cv2.MORPH_OPEN, kernel)
    collagen_mask = cv2.morphologyEx(collagen_mask, cv2.MORPH_CLOSE, kernel)

    # Labeling (0/1 statt 0/255)
    labels = label(collagen_mask > 0)
    filtered_mask = np.zeros_like(collagen_mask)

    for region in regionprops(labels):
        if region.major_axis_length >= min_length and region.area >= min_area:
            filtered_mask[labels == region.label] = 255

    collagen_mask = filtered_mask
    cm = collagen_mask > 0

    # -------------------------
    # Hue-Klassifikation
    # -------------------------
    red_mask = (((h >= 0) & (h <= red_max)) | ((h >= 170) & (h <= 179))) & cm
    orange_mask = ((h >= orange_low) & (h <= orange_high)) & cm
    green_mask = ((h >= green_low) & (h <= green_high)) & cm

    # Quantifizierung
    red_px = red_mask.sum()
    green_px = green_mask.sum()

    total_classified = red_px + green_px
    red_rel = 100 * red_px / (total_classified + 1e-6)
    green_rel = 100 * green_px / (total_classified + 1e-6)

    total_area = cm.sum()

    # Overlay erstellen
    overlay = img.copy()
    overlay = overlay.astype(np.uint8)
    overlay[red_mask] = [255, 0, 0]
    overlay[orange_mask] = [255, 165, 0]
    overlay[green_mask] = [0, 255, 0]

    return {
        "Image": file.name,
        "Total Collagen Area (px)": total_area,
        "Collagen I (red %)": red_rel,
        "Collagen III (green %)": green_rel,
        "overlay": overlay,
        "mask": collagen_mask
    }

# -------------------------
# Main
# -------------------------
results = []

if uploaded:
    for f in uploaded:
        f.seek(0)
        results.append(analyze_image(f))

    df = pd.DataFrame(results).drop(columns=["overlay", "mask"])
    st.subheader("📄 Ergebnisse (Rot/Grün Relation)")
    st.dataframe(df)

    st.download_button(
        "📥 CSV herunterladen",
        df.to_csv(index=False).encode("utf-8"),
        "psr_collagen_results.csv"
    )

    st.subheader("🔍 Qualitätskontrolle")
    for r in results:
        st.markdown(f"### {r['Image']}")
        st.image(r["mask"], caption="Gesamt-Kollagen-Maske inkl. feiner Fasern")
        st.image(
            r["overlay"],
            caption="Overlay: Rot (I) · Orange (I+III, Remis) · Grün (III)"
        )
else:
    st.info("Bitte PSR-Bilder hochladen.")
