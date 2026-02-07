# QR Code Detector (CPU)
A lightweight Streamlit app that counts and decodes multiple QR codes from a single image using OpenCV's `QRCodeDetector` (CPU-friendly, no GPU required).

## Why this model?
OpenCV's built-in `QRCodeDetector` is fast, runs on CPU, and supports detecting and decoding multiple QR codes in one pass. It avoids heavyweight deep learning models while still providing reliable results for typical QR images.

## Features
- Upload an image with multiple QR codes.
- Count all detected QR codes.
- Decode and list the embedded data.
- Visualize bounding boxes and labels.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```
