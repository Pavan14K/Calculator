"""Streamlit UI for CPU-based QR code detection and decoding."""
from __future__ import annotations

from typing import List

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from qr_detector import QRCodeResult, count_qr_codes, detect_qr_codes, draw_qr_boxes


def load_image(image_file: Image.Image) -> np.ndarray:
    """Convert a PIL image to an OpenCV-compatible BGR array.

    Args:
        image_file: The uploaded PIL image.

    Returns:
        The image as a NumPy array in BGR format.
    """
    rgb_image = np.array(image_file.convert("RGB"))
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)


def format_results(results: List[QRCodeResult]) -> List[str]:
    """Format decoded QR code results for display.

    Args:
        results: List of QRCodeResult items.

    Returns:
        A list of human-readable strings for each decoded QR code.
    """
    return [f"{idx}. {result.data}" for idx, result in enumerate(results, start=1)]


def main() -> None:
    """Render the Streamlit interface."""
    st.set_page_config(page_title="QR Code Detector", layout="wide")
    st.title("CPU QR Code Detector")
    st.write(
        "Upload an image with multiple QR codes. The app will count and decode them "
        "using OpenCV's QRCodeDetector (CPU-friendly)."
    )

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if not uploaded_file:
        st.info("Please upload an image to begin.")
        return

    image = Image.open(uploaded_file)
    bgr_image = load_image(image)
    results = detect_qr_codes(bgr_image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Detected")
        annotated = draw_qr_boxes(bgr_image, results)
        rgb_annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(rgb_annotated, use_column_width=True)

    st.metric("QR Codes Found", count_qr_codes(results))

    if results:
        st.subheader("Decoded Data")
        st.write(format_results(results))
    else:
        st.warning("No QR codes detected. Try a clearer image.")


if __name__ == "__main__":
    main()
