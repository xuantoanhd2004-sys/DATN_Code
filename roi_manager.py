from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import config
from models import *
from utils import *
ROI_POINTS: list[tuple[int, int]] = []
ROI_EDIT_POINTS: list[tuple[int, int]] = []

# Làm sạch và chuẩn hóa danh sách tọa độ ROI thành các cặp (x, y) hợp lệ.
def _clean_roi_points(points: Any) -> list[tuple[int, int]]:
    cleaned: list[tuple[int, int]] = []
    if not isinstance(points, list):
        return cleaned
    for item in points:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            cleaned.append((int(item[0]), int(item[1])))
        except Exception:
            continue
    return cleaned


# Trả về đường dẫn file cấu hình ROI trên đĩa.
def roi_config_path(paths: dict[str, Path] | None = None) -> Path:
    if paths is not None and "config" in paths:
        return paths["config"] / config.ROI_CONFIG_FILENAME
    return app_root() / "config" / config.ROI_CONFIG_FILENAME


# Nạp cấu hình vùng giám sát ROI (đơn vị phần trăm màn hình) từ file config.
def load_roi_config(paths: dict[str, Path], logger=None) -> None:
    config.ROI_ENABLED = False
    ROI_POINTS.clear()
    config.ROI_EDIT_MODE = False
    ROI_EDIT_POINTS.clear()

    path = roi_config_path(paths)
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        points = _clean_roi_points(data.get("points", []))
        ROI_POINTS.clear()
        if len(points) >= 3:
            ROI_POINTS.extend(points)
        config.ROI_ENABLED = bool(data.get("roi_enabled", False)) and len(ROI_POINTS) >= 3
        if logger is not None:
            logger.remember("ROI loaded" if config.ROI_ENABLED else "ROI config loaded, ROI disabled")
    except Exception:
        config.ROI_ENABLED = False
        ROI_POINTS.clear()
        if logger is not None:
            logger.remember("Cảnh báo: lỗi đọc roi_config.json, ROI tắt")


# Lưu điểm ảnh của vùng ROI vào đĩa cứng (lưu dưới dạng phần trăm 0.0-1.0
# để tương thích khi đổi độ phân giải camera).
def save_roi_config(paths: dict[str, Path] | None = None) -> None:
    path = roi_config_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    enabled = bool(config.ROI_ENABLED and len(ROI_POINTS) >= 3)
    payload = {
        "roi_enabled": enabled,
        "points": [[int(x), int(y)] for x, y in ROI_POINTS],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# Kích hoạt chế độ vẽ vùng ROI thủ công bằng chuột trên màn hình camera.
def start_roi_edit(logger=None) -> None:
    config.ROI_EDIT_MODE = True
    ROI_EDIT_POINTS.clear()
    if logger is not None:
        logger.remember("ROI edit mode")


# Hoàn tất việc vẽ ROI: Lưu các điểm đã vẽ vào config, xóa chế độ edit.
def save_roi_edit(paths: dict[str, Path], logger=None) -> None:
    if len(ROI_EDIT_POINTS) < 3:
        if logger is not None:
            logger.remember("ROI cần ít nhất 3 điểm")
        return
    ROI_POINTS.clear()
    ROI_POINTS.extend(ROI_EDIT_POINTS)
    config.ROI_ENABLED = True
    config.ROI_EDIT_MODE = False
    ROI_EDIT_POINTS.clear()
    save_roi_config(paths)
    if logger is not None:
        logger.remember("ROI saved")


# Hủy bỏ việc vẽ ROI, không lưu gì cả.
def cancel_roi_edit(logger=None) -> None:
    config.ROI_EDIT_MODE = False
    ROI_EDIT_POINTS.clear()
    if logger is not None:
        logger.remember("ROI edit canceled")


# Xóa toàn bộ vùng ROI đã vẽ và tắt chức năng ROI.
def clear_roi(paths: dict[str, Path], logger=None) -> None:
    config.ROI_ENABLED = False
    ROI_POINTS.clear()
    config.ROI_EDIT_MODE = False
    ROI_EDIT_POINTS.clear()
    save_roi_config(paths)
    if logger is not None:
        logger.remember("ROI cleared")


# Bật/tắt chức năng giám sát vùng ROI (toggle on/off).
def toggle_roi(paths: dict[str, Path], logger=None) -> None:

    if config.ROI_ENABLED:
        config.ROI_ENABLED = False
        save_roi_config(paths)
        if logger is not None:
            logger.remember("ROI disabled")
        return
    if len(ROI_POINTS) < 3:
        if logger is not None:
            logger.remember("Chưa có ROI. Bấm E để vẽ ROI.")
        return
    config.ROI_ENABLED = True
    save_roi_config(paths)
    if logger is not None:
        logger.remember("ROI enabled")


# Chuyển đổi tọa độ ROI từ phần trăm sang tọa độ pixel thực tế của khung hình.
def roi_polygon_pixels(frame_shape) -> list[tuple[int, int]]:
    h, w = frame_shape[:2]
    if not config.ROI_ENABLED or len(ROI_POINTS) < 3:
        return []
    clipped = []
    for x, y in ROI_POINTS:
        clipped.append((int(clamp(x, 0, max(0, w - 1))), int(clamp(y, 0, max(0, h - 1)))))
    return clipped


# Thuật toán Point-in-Polygon: kiểm tra xem điểm (tọa độ trung tâm người)
# có nằm trong đa giác ROI hay không dựa vào thư viện cv2.pointPolygonTest.
def point_in_polygon(point: tuple[float, float], polygon: list[tuple[int, int]]) -> bool:
    if len(polygon) < 3:
        return False
    pts = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(pts, point, False) >= 0


# Callback xử lý sự kiện click chuột trên cửa sổ OpenCV khi đang vẽ ROI.
def roi_mouse_callback(event, x, y, flags, param) -> None:
    del flags, param

    if event == cv2.EVENT_LBUTTONDOWN and config.ROI_EDIT_MODE:
        ROI_EDIT_POINTS.append((int(x), int(y)))


# Cập nhật trạng thái ROI (trong/ngoài vùng) cho tất cả mục tiêu đang theo dõi.
def refresh_target_roi_status(manager) -> None:
    polygon = list(ROI_POINTS) if config.ROI_ENABLED and len(ROI_POINTS) >= 3 else []
    for target in manager.targets.values():
        if not polygon:
            target.roi_status = "roi_disabled"
            continue
        cx, cy = bbox_center(target.bbox)
        target.roi_status = "inside_roi" if point_in_polygon((cx, cy), polygon) else "outside_roi"


