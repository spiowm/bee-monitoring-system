"""
TEM-inspired motion intensity computation for individual bee crops.

Замість побудови повного TEM-зображення (що потребує перетренування моделі),
обчислюємо числовий motion score — частку пікселів що змінились у crop-і
бджоли між послідовними кадрами.

Це дає додатковий сигнал для розрізнення:
  - Fanning:     high motion (крила!) + low displacement
  - Washboarding: medium motion + rhythmic ZCR
  - Idle/Unknown: low motion + low displacement
"""
import cv2
import numpy as np


def compute_crop_motion(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    bbox: tuple,
    threshold: int = 15,
) -> float:
    """
    Обчислює TEM-подібний motion intensity для crop бджоли.

    Parameters
    ----------
    prev_frame : попередній кадр (BGR)
    curr_frame : поточний кадр (BGR)
    bbox       : (x1, y1, x2, y2) bounding box бджоли
    threshold  : поріг різниці пікселів для бінарізації

    Returns
    -------
    float: частка змінених пікселів (0.0 – 1.0)
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = prev_frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 - x1 < 4 or y2 - y1 < 4:
        return 0.0

    prev_crop = cv2.cvtColor(prev_frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    curr_crop = cv2.cvtColor(curr_frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(prev_crop, curr_crop)
    _, thresh_img = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    return float(np.sum(thresh_img > 0)) / (thresh_img.size + 1e-6)
