"""Utilities for detecting and decoding QR codes on CPU."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class QRCodeResult:
    """Represents a decoded QR code and its bounding box."""

    data: str
    points: np.ndarray


def detect_qr_codes(image: np.ndarray) -> List[QRCodeResult]:
    """Detect and decode multiple QR codes from a BGR image.

    Args:
        image: The input image in BGR color space.

    Returns:
        A list of QRCodeResult objects containing decoded data and points.
    """
    detector = cv2.QRCodeDetector()
    decoded_texts: Sequence[str]
    points: np.ndarray | None

    decoded_texts, points, _ = detector.detectAndDecodeMulti(image)

    if points is None or len(decoded_texts) == 0:
        return []

    results: List[QRCodeResult] = []
    for text, box in zip(decoded_texts, points):
        if text:
            results.append(QRCodeResult(data=text, points=box))

    return results


def draw_qr_boxes(image: np.ndarray, results: Sequence[QRCodeResult]) -> np.ndarray:
    """Draw QR bounding boxes and labels on a copy of the image.

    Args:
        image: The input image in BGR color space.
        results: Iterable of QRCodeResult values to draw.

    Returns:
        A copy of the image with drawn bounding boxes and labels.
    """
    output = image.copy()
    for idx, result in enumerate(results, start=1):
        pts = result.points.astype(int)
        cv2.polylines(output, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        label = f"{idx}: {result.data}"
        text_origin = tuple(pts[0])
        cv2.putText(
            output,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return output


def count_qr_codes(results: Sequence[QRCodeResult]) -> int:
    """Count the decoded QR codes.

    Args:
        results: Iterable of QRCodeResult values.

    Returns:
        The number of decoded QR codes.
    """
    return len(results)
