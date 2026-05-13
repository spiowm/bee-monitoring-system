import numpy as np

def get_orientation_vector(keypoints_xy):
    """
    keypoints_xy: np.array shape (2, 2) — [[x_head, y_head], [x_tail, y_tail]]
    Повертає нормалізований вектор (dx, dy) від жала до голови.
    Повертає None якщо норма вектора < 1e-6 (вироджений випадок).
    """
    if keypoints_xy is None or len(keypoints_xy) < 2:
        return None

    keypoints_xy = np.asarray(keypoints_xy, dtype=np.float32)
    head = keypoints_xy[0]
    tail = keypoints_xy[1]
    
    vec = head - tail
    norm = np.linalg.norm(vec)
    
    if norm < 1e-6:
        return None
        
    return vec / norm

def should_count_crossing(track_direction_vec, orientation_vec, threshold_deg=60.0):
    """
    track_direction_vec: вектор руху з останніх N позицій треку
    orientation_vec: з get_orientation_vector() або None
    Якщо None → fallback, повертає True (рахувати без фільтрації).
    Якщо кут між векторами < threshold_deg → True (підтверджений перетин).
    """
    if orientation_vec is None:
        return True
        
    if track_direction_vec is None:
        return True
        
    track_norm = np.linalg.norm(track_direction_vec)
    if track_norm < 1e-6:
        return True
        
    track_dir = track_direction_vec / track_norm
    
    dot_product = np.dot(track_dir, orientation_vec)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg <= threshold_deg

def vector_to_entrance(bee_xy, ramp_kpts):
    """
    Одиничний вектор від центру бджоли до найближчої точки лінії входу,
    яка задана першими двома точками рампи: ramp_kpts[0] → ramp_kpts[1].
    Повертає None, якщо рамп невідомий або вектор вироджений.
    """
    if ramp_kpts is None or len(ramp_kpts) < 2 or bee_xy is None:
        return None

    p = np.asarray(bee_xy, dtype=np.float32)
    a = np.asarray(ramp_kpts[0][:2], dtype=np.float32)
    b = np.asarray(ramp_kpts[1][:2], dtype=np.float32)
    ab = b - a
    ab_len_sq = float(np.dot(ab, ab))
    if ab_len_sq < 1e-6:
        return None

    t = float(np.dot(p - a, ab)) / ab_len_sq
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    vec = proj - p
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return None
    return vec / norm


def aligned(vec_a, vec_b, max_angle_deg: float) -> bool:
    """Чи кут між нормалізованими векторами <= max_angle_deg. None → fallback True."""
    if vec_a is None or vec_b is None:
        return True
    dot = float(np.clip(np.dot(vec_a, vec_b), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(dot)))
    return angle_deg <= max_angle_deg


def aligned_strict(vec_a, vec_b, max_angle_deg: float) -> bool:
    """Як aligned(), але повертає False коли один з векторів None (без fallback)."""
    if vec_a is None or vec_b is None:
        return False
    dot = float(np.clip(np.dot(vec_a, vec_b), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(dot)))
    return angle_deg <= max_angle_deg


def get_angular_error(pred_vec, gt_vec):
    """
    Для evaluation — кутова похибка між передбаченим і GT вектором.
    Повертає min(angle, 180-angle) в градусах (симетрія голова/жало).
    """
    if pred_vec is None or gt_vec is None:
        return None
        
    dot_product = np.dot(pred_vec, gt_vec)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    
    return min(angle_deg, 180.0 - angle_deg)
