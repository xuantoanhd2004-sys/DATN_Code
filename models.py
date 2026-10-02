from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
# Cấu trúc dữ liệu chứa kết quả thô trả về từ AI Detector sau mỗi frame.
@dataclass
class RawDetection:
    object_id: int | None
    bbox: tuple[int, int, int, int]
    class_name: str
    confidence: float
    analysis: dict[str, Any] = field(default_factory=dict)


# Cấu trúc dữ liệu chứa điều kiện thời tiết (gió, mưa, nhiệt độ) lấy từ API.
@dataclass
class EnvironmentReading:
    source: str
    location_name: str
    temperature_c: float | None
    humidity_percent: float | None
    wind_speed_ms: float | None
    rain_probability_1h: float | None
    precipitation_1h_mm: float | None
    rain_probability_3h: float | None
    precipitation_3h_mm: float | None
    weather_code: int | None
    assessment: str
    recommendation: str
    status_text: str
    updated_text: str
    has_data: bool
    is_stale: bool = False


# Cấu trúc dữ liệu chứa trạng thái phần cứng và nhiệm vụ của hệ thống.
@dataclass
class TelemetrySnapshot:
    mission_state: str
    laptop_battery_text: str
    laptop_charging_text: str
    internet_status: str
    fps: float
    gps_connected: bool
    gps_lat: float | None
    gps_lon: float | None
    altitude_m: float | None
    heading_deg: float | None
    map_status: str
    victim_position_text: str
    drone_battery_text: str
    timestamp_text: str


# Cấu trúc dữ liệu lõi lưu trữ trạng thái TOÀN DIỆN của một mục tiêu đang theo dõi.
# Chứa lịch sử di chuyển, điểm Danger Score, quyết định của Operator và thông tin ROI.
@dataclass
class TargetState:
    object_id: int
    bbox: tuple[int, int, int, int]
    class_name: str
    confidence: float
    first_seen: float
    last_seen: float
    
    stable_class: str = "unknown"
    posture_class: str = "unknown"
    stable_posture_class: str = "unknown"
    rescue_victim_evidence: float = 0.0
    posture_status: str = "unknown"
    
    motion_status: str = "stable"
    roi_status: str = "roi_disabled"
    roi_score: int = 0
    roi_reason: str = "ROI tắt"
    
    score_breakdown: dict[str, int] = field(default_factory=dict)
    score_reasons: list[str] = field(default_factory=list)
    raw_rescue_score: int = 0
    rescue_score: int = 0
    display_status: str = "NORMAL"
    rescue_label: str = "BINH_THUONG"
    display_label: str = "BÌNH THƯỜNG"

    analysis_status: str = "NORMAL"
    final_status: str = ""
    final_priority: int = 0
    suggested_command: str = "HOVER"

    pose_state: str = "BINH_THUONG"
    pose_reasons: list[str] = field(default_factory=list)
    pose_flags: dict[str, bool] = field(default_factory=dict)
    gesture_status: str = "none"
    gesture_candidate: str = "none"
    gesture_candidate_since: float = 0.0
    gesture_last_detected_time: float = 0.0

    detector_source: str = "unknown"
    operator_decision: str | None = None
    suppress_alert_until: float = 0.0
    track_more_until: float = 0.0
    last_auto_log: float = 0.0

    latest_crop: np.ndarray | None = None
    center_history: deque = field(default_factory=lambda: deque(maxlen=240))
    class_history: deque = field(default_factory=lambda: deque(maxlen=30))
    posture_class_history: deque = field(default_factory=lambda: deque(maxlen=30))
    posture_status_history: deque = field(default_factory=lambda: deque(maxlen=30))
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=30))
    score_history: deque = field(default_factory=lambda: deque(maxlen=30))
    raw_status_history: deque = field(default_factory=lambda: deque(maxlen=30))
    left_wrist_history: deque = field(default_factory=lambda: deque(maxlen=120))
    right_wrist_history: deque = field(default_factory=lambda: deque(maxlen=120))
    
    kps_xy: np.ndarray | None = None
    kps_conf: np.ndarray | None = None
    
    # Cắt ảnh crop của mục tiêu từ frame gốc (có thêm viền margin pixel).
    def crop_from(self, frame: np.ndarray, margin: int = 8) -> np.ndarray | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.bbox
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()


