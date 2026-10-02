from __future__ import annotations

import base64
import csv
import html
import os
import time
from pathlib import Path

import config
# Xuất báo cáo HTML sau mỗi chuyến bay, liệt kê lại toàn bộ log DANGER
# và nhúng video/ảnh bằng chứng để phi công xem lại dễ dàng.
class ReportGenerator:
    # Khởi tạo đối tượng xuất báo cáo HTML từ file log CSV.
    def __init__(self, paths: dict[str, Path], root: Path) -> None:
        self.paths = paths
        self.root = root
        self.csv_path = paths["logs"] / "rescue_events.csv"
        self.reports_dir = paths["reports"]

    # Hàm sinh báo cáo: Đọc CSV lịch sử, lọc các sự kiện quan trọng,
    # tìm file media tương ứng và xuất ra thư mục báo cáo bằng HTML/CSS tĩnh.
    def generate(self) -> Path:
        rows = self._filter_report_rows(self._read_rows())
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"rescue_report_{timestamp}.html"
        report_path.write_text(self._build_html(rows, report_path), encoding="utf-8")
        return report_path

    # Đọc dữ liệu thô từ file CSV dưới dạng danh sách dict.
    def _read_rows(self) -> list[dict[str, str]]:
        if not self.csv_path.exists():
            return []
        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    # Lọc bỏ các cảnh báo nhiễu, giữ lại sự kiện quan trọng (Operator quyết định hoặc Danger có ý nghĩa).
    def _filter_report_rows(self, rows: list[dict[str, str]]) -> list[dict[str, str]]:
        filtered: list[dict[str, str]] = []
        last_auto_danger_by_id: dict[str, tuple[float | None, int]] = {}
        operator_events = {"OPERATOR_CONFIRMED", "OPERATOR_FALSE_ALARM", "OPERATOR_TRACK_MORE"}

        for source_row in rows:
            row = dict(source_row)
            event_type = self._normalized_event_type(row.get("event_type", ""))
            row["event_type"] = event_type

            if event_type in operator_events:
                filtered.append(row)
                continue
            if event_type == "AUTO_WARNING":
                filtered.append(row)
                continue
            if event_type != "AUTO_DANGER":
                continue
            if not self._is_significant_auto_danger(row):
                continue

            object_id = str(row.get("object_id", "") or "")
            row_time = self._row_epoch(row)
            score = self._row_score(row)
            previous = last_auto_danger_by_id.get(object_id)
            if previous is None:
                filtered.append(row)
                last_auto_danger_by_id[object_id] = (row_time, score)
                continue

            previous_time, previous_score = previous
            elapsed = None if row_time is None or previous_time is None else row_time - previous_time
            score_jump = score - previous_score >= config.AUTO_DANGER_SCORE_DELTA
            cooldown_ready = elapsed is None or elapsed >= config.AUTO_DANGER_LOG_COOLDOWN_SECONDS
            if score_jump or cooldown_ready:
                filtered.append(row)
                last_auto_danger_by_id[object_id] = (row_time, score)

        return filtered

    # Chuẩn hóa tên sự kiện (ví dụ: OPERATOR_CONFIRM -> OPERATOR_CONFIRMED).
    @staticmethod
    def _normalized_event_type(event_type: str) -> str:
        event_type = str(event_type or "").strip()
        aliases = {
            "OPERATOR_CONFIRM": "OPERATOR_CONFIRMED",
        }
        return aliases.get(event_type, event_type)

    # Chuyển đổi timestamp dạng chuỗi sang Epoch time (giây).
    @staticmethod
    def _row_epoch(row: dict[str, str]) -> float | None:
        try:
            return time.mktime(time.strptime(str(row.get("timestamp", ""))[:19], "%Y-%m-%d %H:%M:%S"))
        except Exception:
            return None

    # Lấy điểm số cứu hộ từ dữ liệu CSV.
    @staticmethod
    def _row_score(row: dict[str, str]) -> int:
        try:
            return int(float(row.get("ai_rescue_score") or row.get("rescue_score") or 0))
        except Exception:
            return 0

    # Đánh giá xem một cảnh báo tự động (AUTO_DANGER) có chứa dấu hiệu rõ ràng hay không.
    @staticmethod
    def _is_significant_auto_danger(row: dict[str, str]) -> bool:
        yolo_class = str(row.get("yolo_class", "") or "")
        posture = str(row.get("posture_status", "") or "")
        motion = str(row.get("motion_status", "") or "")
        gesture = str(row.get("gesture_status", "") or "")
        roi = str(row.get("roi_status", "") or "")
        pose_state = str(row.get("pose_state", "") or "")
        pose_reasons = str(row.get("pose_reasons", "") or "").strip()
        rescue_label = str(row.get("rescue_label", "") or "")
        display_label = str(row.get("display_label", "") or "")
        if yolo_class == "victim_person":
            return True
        if posture == "lying_or_fallen":
            return True
        if gesture in {"waving", "distress_signal", "two_hands_raised"}:
            return True
        if rescue_label in {"NGUY_HIEM", "NGHI_NAN_NHAN"} or display_label in {"NGHI NẠN NHÂN", "NGUY HIỂM"}:
            return True
        if pose_state and pose_state not in {"BINH_THUONG", "normal", "NORMAL"}:
            return True
        if pose_reasons:
            return True
        if motion == "dangerous_motionless" and (yolo_class == "victim_person" or posture == "lying_or_fallen"):
            return True
        if roi == "inside_roi" and (yolo_class == "victim_person" or posture == "lying_or_fallen" or pose_state not in {"", "BINH_THUONG"}):
            return True
        return False

    # Lắp ráp HTML hoàn chỉnh từ dữ liệu lịch sử và các thẻ sự kiện.
    def _build_html(self, rows: list[dict[str, str]], report_path: Path) -> str:
        total = len(rows)
        danger = sum(1 for r in rows if r.get("event_type") == "AUTO_DANGER" or r.get("display_status") == "DANGER" or r.get("status") == "DANGER")
        confirmed = sum(1 for r in rows if r.get("operator_decision") == "CONFIRMED" or r.get("final_status") == "CONFIRMED_VICTIM")
        false_alarm = sum(1 for r in rows if r.get("operator_decision") == "FALSE_ALARM")
        cards = "\n".join(self._event_card(row, report_path) for row in rows) or "<p>Chưa có sự kiện.</p>"
        return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <title>Báo cáo nhiệm vụ tìm kiếm cứu nạn</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; background: #f5f7fb; color: #17202a; }}
    h1 {{ margin-bottom: 4px; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 12px; margin: 18px 0; }}
    .metric, .event {{ background: white; border: 1px solid #d9e1ec; border-radius: 6px; padding: 14px; }}
    .metric b {{ display: block; font-size: 24px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
    th, td {{ border-bottom: 1px solid #e1e6ef; padding: 6px; text-align: left; vertical-align: top; }}
    img {{ max-width: 260px; max-height: 180px; border: 1px solid #ccd4df; margin: 6px 10px 6px 0; }}
    video {{ max-width: 320px; border: 1px solid #ccd4df; display: block; margin-top: 6px; }}
    .media {{ display: flex; flex-wrap: wrap; gap: 12px; }}
    .muted {{ color: #667085; }}
  </style>
</head>
<body>
  <h1>BÁO CÁO NHIỆM VỤ TÌM KIẾM CỨU NẠN</h1>
  <div class="muted">Thời gian tạo báo cáo: {html.escape(time.strftime("%Y-%m-%d %H:%M:%S"))}</div>
  <div class="summary">
    <div class="metric"><b>{total}</b>Tổng số sự kiện</div>
    <div class="metric"><b>{danger}</b>Cảnh báo DANGER</div>
    <div class="metric"><b>{confirmed}</b>CONFIRMED_VICTIM</div>
    <div class="metric"><b>{false_alarm}</b>FALSE_ALARM</div>
  </div>
  {cards}
</body>
</html>"""

    # Render nội dung chi tiết của một sự kiện (bảng thông số và ảnh/video) vào 1 Card HTML.
    def _event_card(self, row: dict[str, str], report_path: Path) -> str:
        fields = [
            "event_id",
            "timestamp",
            "object_id",
            "event_type",
            "yolo_class",
            "yolo_conf",
            "rescue_label",
            "display_label",
            "score_yolo",
            "score_posture",
            "score_motion",
            "score_gesture",
            "score_roi",
            "score_total",
            "score_reason",
            "rescue_score",
            "ai_rescue_score",
            "display_status",
            "analysis_status",
            "ai_status",
            "final_status",
            "final_priority",
            "posture_status",
            "motion_status",
            "gesture_status",
            "roi_status",
            "weather_source",
            "weather_location",
            "weather_temperature_c",
            "weather_humidity_percent",
            "weather_wind_speed_ms",
            "weather_rain_probability_1h",
            "weather_precipitation_1h_mm",
            "weather_rain_probability_3h",
            "weather_precipitation_3h_mm",
            "weather_assessment",
            "weather_recommendation",
            "operator_decision",
        ]
        table = "\n".join(
            f"<tr><th>{html.escape(field)}</th><td>{html.escape(str(row.get(field, '') or ''))}</td></tr>"
            for field in fields
        )
        score_detail = (
            "<h3>Chi tiết điểm nguy cơ</h3>"
            "<table>"
            f"<tr><th>YOLO</th><td>+{html.escape(str(row.get('score_yolo', '') or '0'))}</td></tr>"
            f"<tr><th>Tư thế</th><td>+{html.escape(str(row.get('score_posture', '') or '0'))}</td></tr>"
            f"<tr><th>Chuyển động</th><td>+{html.escape(str(row.get('score_motion', '') or '0'))}</td></tr>"
            f"<tr><th>Cử chỉ</th><td>+{html.escape(str(row.get('score_gesture', '') or '0'))}</td></tr>"
            f"<tr><th>ROI</th><td>+{html.escape(str(row.get('score_roi', '') or '0'))}</td></tr>"
            f"<tr><th>Tổng</th><td>{html.escape(str(row.get('score_total', '') or row.get('rescue_score', '') or '0'))}/100</td></tr>"
            f"<tr><th>Lý do</th><td>{html.escape(str(row.get('score_reason', '') or ''))}</td></tr>"
            "</table>"
        )
        media = self._media_html("Ảnh full", row.get("image_full_path", ""), report_path, "image")
        media += self._media_html("Ảnh crop", row.get("image_crop_path", ""), report_path, "image")
        media += self._media_html("Video replay", row.get("video_replay_path", ""), report_path, "video")
        return f"""<section class="event">
  <h2>{html.escape(row.get("event_id", "Sự kiện"))}</h2>
  {score_detail}
  <table>{table}</table>
  <div class="media">{media}</div>
</section>"""

    # Trả về thẻ HTML nhúng ảnh (Base64) hoặc video Replay.
    def _media_html(self, label: str, path_text: str, report_path: Path, kind: str) -> str:
        if not path_text:
            return self._missing_media_html(label, kind)
        path = Path(path_text)
        if kind == "video":
            video_path = self._valid_video_path(path)
            if video_path is None:
                return self._missing_media_html(label, kind)
            rel = os.path.relpath(video_path, start=report_path.parent).replace("\\", "/")
            return f'<div><b>{html.escape(label)}</b><video controls src="{html.escape(rel)}"></video></div>'
        if not path.exists():
            return self._missing_media_html(label, kind)
        try:
            img_bytes = path.read_bytes()
            ext = path.suffix.lower().lstrip(".")
            mime = "jpeg" if ext in ("jpg", "jpeg") else ext
            b64 = base64.b64encode(img_bytes).decode("ascii")
            data_uri = f"data:image/{mime};base64,{b64}"
            return f'<div><b>{html.escape(label)}</b><br><img src="{data_uri}" alt="{html.escape(label)}"></div>'
        except Exception:
            return self._missing_media_html(label, kind)

    # HTML thông báo khi thiếu ảnh/video bằng chứng.
    @staticmethod
    def _missing_media_html(label: str, kind: str) -> str:
        text = "Không có video replay hợp lệ" if kind == "video" else "Không có"
        return f"<div><b>{html.escape(label)}</b><br>{text}</div>"

    # Tìm đường dẫn file video hợp lệ (thử mp4 trước, nếu thiếu thì thử avi).
    @staticmethod
    def _valid_video_path(path: Path) -> Path | None:
        candidates = [path]
        if path.suffix.lower() == ".mp4":
            candidates.append(path.with_suffix(".avi"))
        for candidate in candidates:
            try:
                if candidate.exists() and candidate.stat().st_size > config.REPLAY_MIN_VALID_BYTES:
                    return candidate
            except OSError:
                continue
        return None


STATUS_COLORS = {
    "NORMAL": (40, 210, 80),
    "WARNING": (0, 210, 255),
    "DANGER": (40, 40, 235),
    "CONFIRMED_VICTIM": (220, 70, 220),
    "FALSE_ALARM": (150, 150, 150),
}


