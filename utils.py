from __future__ import annotations

from pathlib import Path

import config
from models import *
# Trả về đường dẫn gốc của dự án (thư mục cha chứa thư mục code_chinh).
def app_root() -> Path:
    return Path(__file__).resolve().parent.parent


# Tạo cây thư mục lưu trữ: logs, ảnh cảnh báo, video replay, báo cáo, config.
def ensure_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "logs": root / "logs",
        "full": root / "canh_bao_full",
        "crop": root / "canh_bao_crop",
        "video": root / "canh_bao_video",
        "reports": root / "reports",
        "config": root / "config",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


# Giới hạn giá trị trong khoảng [low, high] (cắt trên, cắt dưới).
def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# Tính tọa độ (x, y) của điểm trung tâm Bounding Box.
def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


# Tính chỉ số IoU (Intersection over Union) giữa 2 Bounding Box.
# Dùng để đối khớp (Association) trong thuật toán tracking.
def iou_overlap(box_a, box_b) -> float:
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter = max(0, x_b - x_a) * max(0, y_b - y_a)
    if inter == 0:
        return 0.0
    area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / float(union) if union > 0 else 0.0


# Lấy tọa độ (x, y) của 1 điểm keypoint (Pose) nếu độ tự tin vượt ngưỡng min_conf.
def get_keypoint(kps_xy, kps_conf, index: int, min_conf: float = 0.35):
    if kps_xy is None or kps_conf is None or index >= len(kps_xy):
        return None
    if kps_conf[index] < min_conf:
        return None
    x, y = kps_xy[index]
    return float(x), float(y)


# Đếm số lần đảo chiều di chuyển trong mảng tọa độ (dùng nội bộ cho detect_wrist_waving).
def _direction_changes(values: list[float], min_delta: float = 0.05) -> int:
    if len(values) < 3:
        return 0
    signs: list[int] = []
    anchor = values[0]
    for value in values[1:]:
        delta = value - anchor
        if abs(delta) < min_delta:
            continue
        sign = 1 if delta > 0 else -1
        if not signs or sign != signs[-1]:
            signs.append(sign)
        anchor = value
    return max(0, len(signs) - 1)


# Thuật toán phát hiện hành động "vẫy tay": phân tích lịch sử tọa độ cổ tay,
# tìm số lần đảo chiều hướng di chuyển lên/xuống liên tục trong thời gian ngắn.
def detect_wrist_waving(history, current_time: float) -> bool:
    recent = [item for item in history if current_time - float(item[0]) <= config.WAVING_WINDOW_SECONDS]
    if len(recent) < 4:
        return False
    xs = [float(item[1]) for item in recent]
    ys = [float(item[2]) for item in recent]
    above_values = [bool(item[3]) for item in recent]
    
    above_count = sum(1 for value in above_values if value)
    if above_count / len(recent) < getattr(config, 'WAVING_MIN_ABOVE_RATIO', 0.7):
        return False

    move_x = max(xs) - min(xs)
    move_y = max(ys) - min(ys)
    if max(move_x, move_y) < config.WRIST_MOVE_THRESHOLD:
        return False

    direction_changes = max(_direction_changes(xs, 0.05), _direction_changes(ys, 0.05))
    return direction_changes >= config.WAVING_MIN_DIRECTION_CHANGES


# Chuẩn hóa tên class tư thế về 3 nhãn chuẩn: standing_person / sitting_person / lying_person.
def normalize_posture_class(cls: str) -> str:
    class_name = str(cls or "").lower()
    if class_name in {"standing", "standing_person", "stand", "0"}:
        return "standing_person"
    if class_name in {"sitting", "sitting_person", "sit", "1"}:
        return "sitting_person"
    if class_name in {"lying", "lying_person", "fallen", "fall", "lie", "2"}:
        return "lying_person"
    return "unknown"


# Chuẩn hóa tên class cứu hộ về 2 nhãn: victim / normal_person.
def normalize_rescue_class(cls: str, class_id: int | None = None) -> str:
    class_name = str(cls or "").strip().lower()
    if class_name in {"victim", "injured", "casualty", "nguoi_gap_nan"}:
        return "victim"
    if class_name in {"normal_person", "normal", "person", "nguoi_binh_thuong"}:
        return "normal_person"
    if class_id is not None:
        return "victim" if int(class_id) == 1 else "normal_person"
    return "normal_person"


