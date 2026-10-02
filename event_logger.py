from __future__ import annotations

import csv
import queue
import threading
import time
import unicodedata
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import config
from models import *
# Hệ thống ghi nhật ký sự kiện, lưu trữ bằng chứng (ảnh, video) và quản lý CSV.
class EventLogger:
    # Khởi tạo EventLogger (quản lý lưu ảnh, video, và CSV log).
    def __init__(self, paths: dict[str, Path]) -> None:
        self.paths = paths
        self.events = deque(maxlen=8)
        self.event_history: list[dict[str, str]] = []
        self.log_path = paths["logs"] / "rescue_events.csv"
        self.history_path = paths["logs"] / "event_history.txt"
        self._auto_danger_by_id: dict[int, dict[str, Any]] = {}
        self._saved_media_by_id: dict[int, list[str]] = {}
        self._last_message_time: dict[str, float] = {}
        self._seen_target_ids: set[int] = set()
        self._last_target_status: dict[int, str] = {}
        self._ensure_csv()
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

    # Ghi nhận một thông báo vào bộ nhớ và file text. Có cơ chế chống trùng lặp (dedupe)
    # trong một khoảng thời gian nhất định để tránh spam log.
    def remember(self, message: str, level: str | None = None, dedupe_seconds: float = 1.0, dedupe_key: str | None = None) -> None:
        now = time.time()
        clean_message = self._normalize_message(str(message))
        clean_level = level or self._infer_level(clean_message)
        key = dedupe_key or f"{clean_level}:{clean_message}"
        last = self._last_message_time.get(key, 0.0)
        if dedupe_seconds > 0 and now - last < dedupe_seconds:
            return
        self._last_message_time[key] = now
        time_text = time.strftime("%H:%M:%S")
        entry = {"time": time_text, "level": clean_level, "message": clean_message}
        self.event_history.append(entry)
        self.events.appendleft(f"{time_text} {clean_message}")
        self._append_history_file(entry)

    # Trả về danh sách 8 sự kiện gần nhất (hiển thị trên giao diện).
    def recent_events(self) -> list[str]:
        return list(self.events)

    # Trả về toàn bộ lịch sử sự kiện dưới dạng chuỗi văn bản.
    def formatted_history(self) -> str:
        return "\n".join(self._format_entry(entry) for entry in self.event_history)

    # Theo dõi danh sách mục tiêu: tự động ghi log khi phát hiện mục tiêu mới hoặc thay đổi trạng thái.
    def observe_targets(self, targets: list[TargetState]) -> None:
        for target in targets:
            if target.object_id not in self._seen_target_ids:
                self._seen_target_ids.add(target.object_id)
                self.remember(f"Phát hiện ID {target.object_id}: {target.class_name}.", "INFO", dedupe_seconds=0)
            previous_status = self._last_target_status.get(target.object_id)
            if previous_status is None:
                self._last_target_status[target.object_id] = target.display_status
                continue
            if previous_status != target.display_status and target.display_status in {"WARNING", "DANGER"}:
                level = "DANGER" if target.display_status == "DANGER" else "WARNING"
                self.remember(
                    f"ID {target.object_id}: YOLO {target.class_name}, đánh giá cứu hộ {target.display_label}, AI Score {target.rescue_score}/100.",
                    level,
                    dedupe_key=f"status:{target.object_id}:{target.display_status}",
                    dedupe_seconds=2.0,
                )
            self._last_target_status[target.object_id] = target.display_status

    # Kiểm tra xem một mục tiêu có thỏa mãn điều kiện để TỰ ĐỘNG GHI LOG NGUY HIỂM hay không.
    # Ngăn chặn việc ghi trùng lặp liên tục khi điểm DANGER tăng/giảm nhẹ qua cooldown.
    def should_auto_log_danger(self, target: TargetState, timestamp: float) -> bool:
        state = self._auto_danger_by_id.setdefault(
            target.object_id,
            {
                "last_seen_status": "NORMAL",
                "last_saved_time": None,
                "last_saved_score": None,
                "last_skip_message_time": 0.0,
            },
        )
        previous_status = str(state.get("last_seen_status", "NORMAL") or "NORMAL")
        state["last_seen_status"] = target.display_status
        if target.display_status != "DANGER":
            return False
        if not self._should_save_media_event("AUTO_DANGER", target, "AUTO"):
            return False

        last_saved_time_raw = state.get("last_saved_time")
        last_saved_score_raw = state.get("last_saved_score")
        never_saved = last_saved_time_raw is None
        last_saved_time = float(last_saved_time_raw or 0.0)
        last_saved_score = int(last_saved_score_raw if last_saved_score_raw is not None else -999)
        just_entered_danger = previous_status != "DANGER"
        score_jump = not never_saved and target.rescue_score - last_saved_score >= config.AUTO_DANGER_SCORE_DELTA
        cooldown_ready = timestamp - last_saved_time >= config.AUTO_DANGER_LOG_COOLDOWN_SECONDS

        if not (never_saved or score_jump or (cooldown_ready and (just_entered_danger or target.display_status == "DANGER"))):
            last_skip_message = float(state.get("last_skip_message_time", 0.0) or 0.0)
            if timestamp - last_skip_message >= config.AUTO_DANGER_SKIP_MESSAGE_SECONDS:
                self.remember(f"ID {target.object_id} vẫn DANGER, bỏ qua lưu trùng", "DANGER", dedupe_key=f"skip-danger:{target.object_id}", dedupe_seconds=config.AUTO_DANGER_SKIP_MESSAGE_SECONDS)
                state["last_skip_message_time"] = timestamp
            return False

        state["last_saved_time"] = timestamp
        state["last_saved_score"] = target.rescue_score
        target.last_auto_log = timestamp
        self.remember(
            f"Cảnh báo nguy hiểm: ID {target.object_id} được đánh giá {target.display_label}, YOLO class = {target.class_name}, AI Score {target.rescue_score}/100.",
            "DANGER",
            dedupe_seconds=0,
        )
        return True

    # Kiểm tra sự kiện có đủ điều kiện lưu ảnh/video không (chỉ lưu khi CONFIRMED hoặc DANGER đủ điểm).
    @staticmethod
    def _should_save_media_event(event_type: str, target: TargetState, decision: str) -> bool:
        if event_type == "OPERATOR_CONFIRMED" or decision == "CONFIRMED":
            return True
        if event_type != "AUTO_DANGER":
            return False
        return target.display_status == "DANGER" and target.rescue_score >= config.AUTO_SAVE_MIN_SCORE

    # Lưu thông tin sự kiện nạn nhân vào file CSV, kèm theo lưu 2 ảnh (Full & Crop).
    # Kích hoạt trích xuất video Replay nếu đủ điều kiện.
    def save_target_event(self, event_type: str, target: TargetState, frame, telemetry: TelemetrySnapshot, env: EnvironmentReading, decision: str, replay_buffer=None) -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        event_id = f"{timestamp}_ID{target.object_id}_{event_type}"
        full_path = self.paths["full"] / f"{event_id}_full.jpg"
        crop_path = self.paths["crop"] / f"{event_id}_crop.jpg"
        save_media = self._should_save_media_event(event_type, target, decision)
        full_path_text = ""
        crop_path_text = ""
        video_replay_path = ""
        if save_media:
            event_frame = frame.copy()
            x1, y1, x2, y2 = target.bbox
            color = (220, 70, 220) if target.final_status == "CONFIRMED_VICTIM" else (40, 40, 235) if target.display_status == "DANGER" else (0, 210, 255)
            cv2.rectangle(event_frame, (x1, y1), (x2, y2), color, 3)
            label = f"ID {target.object_id} | {target.class_name} -> {target.display_label} | AI {target.rescue_score}/100"
            if target.operator_decision:
                label += f" | Operator {target.operator_decision}"
            safe_label = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode("ascii")
            cv2.putText(event_frame, safe_label, (max(0, x1), max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            cv2.imwrite(str(full_path), event_frame)
            full_path_text = str(full_path)

            crop = target.latest_crop if target.latest_crop is not None else target.crop_from(frame)
            if crop is not None:
                cv2.imwrite(str(crop_path), crop)
                crop_path_text = str(crop_path)
        if save_media and replay_buffer is not None:
            video_replay_path = replay_buffer.request_save(event_id, event_type, target) or ""
        
        self._remember_saved_media(
            target.object_id,
            [full_path_text, crop_path_text, video_replay_path],
        )
        if full_path_text:
            self.remember(f"Đã lưu ảnh toàn cảnh: {full_path.name}", "SYSTEM")
        if crop_path_text:
            self.remember(f"Đã lưu ảnh crop mục tiêu: {crop_path.name}", "SYSTEM")

        with self.log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames())
            writer.writerow(
                {
                    "event_id": event_id,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "object_id": target.object_id,
                    "event_type": event_type,
                    "yolo_class": target.class_name,
                    "yolo_conf": f"{target.confidence:.3f}",
                    "rescue_label": target.rescue_label,
                    "display_label": target.display_label,
                    "score_yolo": target.score_breakdown.get("yolo", 0),
                    "score_posture": target.score_breakdown.get("posture", 0),
                    "score_motion": target.score_breakdown.get("motion", 0),
                    "score_gesture": target.score_breakdown.get("gesture", 0),
                    "score_roi": target.score_breakdown.get("roi", 0),
                    "score_total": target.score_breakdown.get("total", target.rescue_score),
                    "score_reason": "; ".join(target.score_reasons),
                    "rescue_score": target.rescue_score,
                    "ai_rescue_score": target.rescue_score,
                    "display_status": target.display_status,
                    "analysis_status": target.analysis_status,
                    "ai_status": target.display_status,
                    "final_status": target.final_status,
                    "final_priority": target.final_priority,
                    "status": target.display_status,
                    "posture_status": target.posture_status,
                    "motion_status": target.motion_status,
                    "gesture_status": target.gesture_status,
                    "roi_status": target.roi_status,
                    "pose_state": target.pose_state,
                    "pose_reasons": ";".join(target.pose_reasons),
                    "gps_lat": telemetry.gps_lat or "",
                    "gps_lon": telemetry.gps_lon or "",
                    "image_full_path": full_path_text,
                    "image_crop_path": crop_path_text,
                    "video_replay_path": video_replay_path,
                    "operator_decision": decision,
                    "mission_state": telemetry.mission_state,
                    "weather_source": env.source,
                    "weather_location": env.location_name,
                    "weather_temperature_c": self._fmt_optional(env.temperature_c, 1),
                    "weather_humidity_percent": self._fmt_optional(env.humidity_percent, 0),
                    "weather_wind_speed_ms": self._fmt_optional(env.wind_speed_ms, 1),
                    "weather_rain_probability_1h": self._fmt_optional(env.rain_probability_1h, 0),
                    "weather_precipitation_1h_mm": self._fmt_optional(env.precipitation_1h_mm, 1),
                    "weather_rain_probability_3h": self._fmt_optional(env.rain_probability_3h, 0),
                    "weather_precipitation_3h_mm": self._fmt_optional(env.precipitation_3h_mm, 1),
                    "weather_assessment": env.assessment,
                    "weather_recommendation": env.recommendation,
                }
            )
        return event_id

    # Xóa tất cả ảnh và video đã lưu của mục tiêu bị đánh dấu báo nhầm (False Alarm).
    def remove_saved_media_for_target(self, object_id: int, replay_buffer=None) -> int:
        media_paths = self._saved_media_by_id.pop(object_id, [])
        if replay_buffer is not None and hasattr(replay_buffer, "cancel_target"):
            try:
                media_paths.extend(replay_buffer.cancel_target(object_id, media_paths))
            except Exception:
                pass

        deleted = 0
        unique_paths = self._unique_paths(media_paths)
        for path_text in unique_paths:
            if self._delete_media_file(path_text):
                deleted += 1

        if deleted:
            self.remember(f"Đã xóa 3 file media báo nhầm cho ID {object_id}.", "ACTION", dedupe_seconds=0)
        self._clear_media_csv(object_id)
        return deleted

    # Ghi nhận danh sách file media đã lưu cho một object_id (để xóa khi báo nhầm).
    def _remember_saved_media(self, object_id: int, paths: list[str]) -> None:
        clean_paths = [path for path in paths if path]
        if not clean_paths:
            return
        self._saved_media_by_id.setdefault(object_id, []).extend(clean_paths)

    # Lọc bỏ các đường dẫn bị trùng lặp trong mảng.
    @staticmethod
    def _unique_paths(paths: list[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for raw_path in paths:
            if not raw_path:
                continue
            path_text = str(raw_path)
            if path_text not in seen:
                seen.add(path_text)
                unique.append(path_text)
            path = Path(path_text)
            if path.suffix.lower() == ".mp4":
                fallback = str(path.with_suffix(".avi"))
                if fallback not in seen:
                    seen.add(fallback)
                    unique.append(fallback)
        return unique

    # Xóa 1 file media khỏi đĩa cứng, trả về True nếu thành công.
    @staticmethod
    def _delete_media_file(path_text: str) -> bool:
        try:
            path = Path(path_text)
            if not path.exists():
                return False
            path.unlink()
            return True
        except OSError:
            return False

    # Xóa đường dẫn media trong file CSV của object đã bị False Alarm.
    def _clear_media_csv(self, object_id: int) -> None:
        if not self.log_path.exists():
            return
        try:
            with self.log_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = reader.fieldnames or self._fieldnames()

            changed = False
            for row in rows:
                if str(row.get("object_id", "")) == str(object_id):
                    for field_name in ("image_full_path", "image_crop_path", "video_replay_path"):
                        if row.get(field_name):
                            row[field_name] = ""
                            changed = True

            if not changed:
                return
            with self.log_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        except Exception:
            self.remember(f"Không cập nhật được CSV sau khi báo nhầm ID {object_id}.", "WARNING", dedupe_seconds=0)

    # Ghi một dòng sự kiện vào file text lịch sử.
    def _append_history_file(self, entry: dict[str, str]) -> None:
        try:
            with self.history_path.open("a", encoding="utf-8") as f:
                f.write(self._format_entry(entry) + "\n")
        except Exception:
            pass

    # Format một dòng log thành chuỗi để in ra console.
    @staticmethod
    def _format_entry(entry: dict[str, str]) -> str:
        return f"[{entry.get('time', '')}] [{entry.get('level', 'INFO')}] {entry.get('message', '')}"

    # Tự động suy luận mức độ (level) của log dựa trên nội dung tin nhắn.
    @staticmethod
    def _infer_level(message: str) -> str:
        upper = message.upper()
        if message.startswith(
            (
                "Đã tải cấu hình ROI",
                "Đã tải mô hình",
                "Chưa tải được mô hình",
                "Đã lưu ảnh",
                "Đã lưu video replay",
                "Replay video",
            )
        ):
            return "SYSTEM"
        if message.startswith(
            (
                "Đã bật ROI",
                "Đã tắt ROI",
                "Đang chỉnh ROI",
                "Đã lưu vùng ROI",
                "Đã xóa vùng ROI",
                "Đã hủy chỉnh ROI",
                "Người vận hành",
                "Tiếp tục theo dõi",
                "Đã tạo báo cáo HTML",
                "Đã cập nhật thời tiết",
            )
        ):
            return "ACTION"
        if "DANGER" in upper or "NGUY HIỂM" in upper:
            return "DANGER"
        if "LỖI" in upper or "CẢNH BÁO" in upper or "WARNING" in upper:
            return "WARNING"
        if any(token in upper for token in ("ROI", "XÁC NHẬN", "BÁO NHẦM", "THEO DÕI", "BÁO CÁO", "THỜI TIẾT")):
            return "ACTION"
        if any(token in upper for token in ("MÔ HÌNH", "MODEL", "SAVED", "REPLAY", "ẢNH", "CSV", "HTML")):
            return "SYSTEM"
        return "INFO"

    # Chuẩn hóa thông báo tiếng Anh ngắn gọn thành câu tiếng Việt đầy đủ.
    @staticmethod
    def _normalize_message(message: str) -> str:
        mappings = {
            "ROI loaded": "Đã tải cấu hình ROI. ROI đang bật.",
            "ROI config loaded, ROI disabled": "Đã tải cấu hình ROI. ROI đang tắt.",
            "ROI edit mode": "Đang chỉnh ROI.",
            "ROI edit canceled": "Đã hủy chỉnh ROI.",
            "ROI saved": "Đã lưu vùng ROI.",
            "ROI cleared": "Đã xóa vùng ROI.",
            "ROI disabled": "Đã tắt ROI.",
            "ROI enabled": "Đã bật ROI.",
            "Weather updated": "Đã cập nhật thời tiết từ Open-Meteo.",
            "Report generated": "Đã tạo báo cáo HTML.",
        }
        if message in mappings:
            return mappings[message]
        if message.startswith("MODEL: Pose + Rescue"):
            if "(V6)" in message:
                return "Đã tải mô hình: Pose + Rescue YOLO V6."
            if "(V3)" in message:
                return "Đã tải mô hình: Pose + Rescue YOLO V3."
            return "Đã tải mô hình: Pose + Rescue YOLO."
        if message.startswith("MODEL: Pose only"):
            return "Đã tải mô hình: Pose YOLO. Chưa tải Rescue YOLO."
        if message.startswith("MODEL: chua load"):
            return "Chưa tải được mô hình YOLO."
        return message

    # Format số float an toàn hoặc trả về rỗng nếu None.
    @staticmethod
    def _fmt_optional(value: float | None, digits: int) -> str:
        if value is None:
            return "NO_DATA"
        return f"{value:.{digits}f}"

    # Tạo hoặc cập nhật header của file CSV nếu chưa có hoặc cột bị thiếu.
    def _ensure_csv(self) -> None:
        fieldnames = self._fieldnames()
        if self.log_path.exists():
            try:
                with self.log_path.open("r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    old_fields = list(reader.fieldnames or [])
                    rows = list(reader)
                if old_fields == fieldnames:
                    return
                with self.log_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow({name: row.get(name, "") for name in fieldnames})
                return
            except Exception:
                pass
        with self.log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    # Liệt kê danh sách các cột (headers) cho file CSV.
    @staticmethod
    def _fieldnames() -> list[str]:
        return [
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
            "status",
            "posture_status",
            "motion_status",
            "gesture_status",
            "roi_status",
            "pose_state",
            "pose_reasons",
            "gps_lat",
            "gps_lon",
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
            "image_full_path",
            "image_crop_path",
            "video_replay_path",
            "mission_state",
        ]


# Cửa sổ nhật ký sự kiện dạng GUI (Tkinter) hiển thị lịch sử log.
class EventLogWindow:
    # Khởi tạo cửa sổ hiển thị Log (Console Window trên CV2).
    def __init__(self, logger: EventLogger) -> None:
        self.logger = logger
        self.root = None
        self.text_widget = None

    # Mở cửa sổ log (nếu chưa mở) hoặc làm mới nội dung nếu đang mở.
    def open(self) -> None:
        if self.root is not None:
            self.refresh()
            self._focus()
            return
        try:
            import tkinter as tk
            from tkinter import scrolledtext
        except Exception:
            self.logger.remember("Không mở được cửa sổ nhật ký: thiếu tkinter.", "WARNING")
            return

        self.root = tk.Tk()
        self.root.title("Nhật ký sự kiện - AI GCS")
        self.root.geometry("860x560")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.text_widget = scrolledtext.ScrolledText(self.root, wrap="word", font=("Consolas", 10))
        self.text_widget.pack(fill="both", expand=True, padx=10, pady=(10, 6))

        button_frame = tk.Frame(self.root)
        button_frame.pack(fill="x", padx=10, pady=(0, 10))
        tk.Button(button_frame, text="Làm mới", command=self.refresh, width=12).pack(side="left")
        tk.Button(button_frame, text="Đóng", command=self.close, width=12).pack(side="left", padx=8)

        self.refresh()
        self._focus()

    # Làm mới nội dung cửa sổ log với dữ liệu mới nhất.
    def refresh(self) -> None:
        if self.root is None or self.text_widget is None:
            return
        self.text_widget.configure(state="normal")
        self.text_widget.delete("1.0", "end")
        self.text_widget.insert("1.0", self.logger.formatted_history())
        self.text_widget.configure(state="normal")
        self.text_widget.see("end")

    # Cập nhật Tkinter event loop mà không chặn vòng lặp chính.
    def poll(self) -> None:
        if self.root is None:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except Exception:
            self.root = None
            self.text_widget = None

    # Đóng cửa sổ log.
    def close(self) -> None:
        if self.root is None:
            return
        try:
            self.root.destroy()
        except Exception:
            pass
        self.root = None
        self.text_widget = None

    # Đưa cửa sổ Log lên trên cùng (focus).
    def _focus(self) -> None:
        if self.root is None:
            return
        try:
            self.root.deiconify()
            self.root.attributes("-topmost", True)
            self.root.lift()
            self.root.focus_force()
            self.root.after(500, lambda: self.root is not None and self.root.attributes("-topmost", False))
        except Exception:
            pass


# Quản lý bộ đệm video quay vòng (Ring Buffer) trên RAM để lưu trữ N giây trước
    # và M giây sau khi xảy ra sự kiện phát hiện nạn nhân. Chạy bằng luồng độc lập.
class EventReplayBuffer:
    # Khởi tạo bộ đệm lưu trữ Replay Video ngầm (Background Video Writer).
    def __init__(self, paths: dict[str, Path]) -> None:
        self.enabled = config.SAVE_REPLAY_VIDEO
        self.video_dir = paths["video"]
        self.buffer: deque[tuple[float, np.ndarray]] = deque()
        self.active: list[dict[str, Any]] = []
        self.completed: queue.Queue[tuple[int, str, bool]] = queue.Queue()
        self._save_queue: queue.Queue[tuple[Path, list[np.ndarray], int, float] | None] = queue.Queue()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._last_sample_time = 0.0
        self._last_saved_by_id: dict[int, float] = {}
        self._canceled_paths: set[str] = set()
        self._thread: threading.Thread | None = None
        if self.enabled:
            self._thread = threading.Thread(target=self._writer_loop, name="replay-writer", daemon=True)
            self._thread.start()

    # Thêm frame vào bộ đệm vòng và kiểm tra các task replay đang chạy.
    def add_frame(self, frame, timestamp: float) -> None:
        if not self.enabled or self._thread is None:
            return
        sample_interval = 1.0 / max(1, config.REPLAY_FPS)
        if timestamp - self._last_sample_time < sample_interval:
            return
        self._last_sample_time = timestamp
        frame_copy = frame.copy()
        ready: list[dict[str, Any]] = []
        with self._lock:
            self.buffer.append((timestamp, frame_copy))
            cutoff = timestamp - max(1, config.REPLAY_PRE_SECONDS + 1)
            while self.buffer and self.buffer[0][0] < cutoff:
                self.buffer.popleft()
            for task in list(self.active):
                task["frames"].append(frame_copy)
                if timestamp >= task["end_time"]:
                    real_fps = len(task["frames"]) / max(1.0, timestamp - task["start_time"])
                    task["fps"] = real_fps
                    ready.append(task)
                    self.active.remove(task)
        for task in ready:
            self._queue_completed_task(task)

    # Yêu cầu lưu một đoạn video Replay cho sự kiện nguy hiểm hiện tại.
    # Nếu đang trong cooldown, yêu cầu sẽ bị từ chối để tránh spam ổ cứng.
    def request_save(self, event_id: str, event_type: str, target: TargetState) -> str:
        if not self.enabled:
            return ""
        if event_type not in {"AUTO_DANGER", "OPERATOR_CONFIRMED"}:
            return ""
        now = time.time()
        last = self._last_saved_by_id.get(target.object_id, 0.0)
        if now - last < config.SAVE_COOLDOWN_SECONDS:
            return ""
        self._last_saved_by_id[target.object_id] = now
        path = self.video_dir / f"event_{event_id}.mp4"
        with self._lock:
            items = [item for item in self.buffer if now - item[0] <= config.REPLAY_PRE_SECONDS]
            frames = [item[1] for item in items]
            start_time = items[0][0] if items else now
            self.active.append(
                {
                    "object_id": target.object_id,
                    "path": path,
                    "frames": frames,
                    "start_time": start_time,
                    "end_time": now + config.REPLAY_POST_SECONDS,
                }
            )
        return str(path)

    # Hủy tất cả replay đang chờ của một object (khi đánh dấu False Alarm).
    def cancel_target(self, object_id: int, paths: list[str] | None = None) -> list[str]:
        canceled_paths: list[str] = []

        # Thêm một đường dẫn file media liên quan vào danh sách hủy bỏ.
        def add_path(path_value: Any) -> None:
            if not path_value:
                return
            path = Path(str(path_value))
            for candidate in self._video_path_candidates(path):
                candidate_text = str(candidate)
                self._canceled_paths.add(candidate_text)
                canceled_paths.append(candidate_text)

        with self._lock:
            for task in list(self.active):
                if int(task.get("object_id", -1)) == object_id:
                    self.active.remove(task)
                    add_path(task.get("path"))
            for path_text in paths or []:
                add_path(path_text)
        return canceled_paths

    # Lấy danh sách các replay đã ghi xong (kết quả thành công / thất bại).
    def get_completed(self) -> list[tuple[int, str, bool]]:
        items = []
        while True:
            try:
                items.append(self.completed.get_nowait())
            except queue.Empty:
                break
        return items

    # Dừng toàn bộ hệ thống replay: ghi nốt các task đang chờ, tắt thread.
    def stop(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            for task in list(self.active):
                self._queue_completed_task(task)
            self.active.clear()
        self._stop.set()
        self._save_queue.put(None)
        self._thread.join(timeout=2.0)

    # Đưa một task đã lưu xong vào hàng đợi hoàn thành để luồng chính xử lý.
    def _queue_completed_task(self, task: dict[str, Any]) -> None:
        frames = list(task.get("frames", []))
        if len(frames) < config.REPLAY_MIN_FRAMES:
            self.completed.put((int(task["object_id"]), str(task["path"]), False))
            return
        fps = float(task.get("fps", config.REPLAY_FPS))
        self._save_queue.put((task["path"], frames, int(task["object_id"]), fps))

    # Vòng lặp chính của luồng ghi video (Background Thread).
    def _writer_loop(self) -> None:
        while True:
            task = self._save_queue.get()
            if task is None:
                break
            if len(task) == 3:
                path, frames, object_id = task
                fps = config.REPLAY_FPS
            else:
                path, frames, object_id, fps = task

            if self._is_canceled_path(path):
                continue
            try:
                saved_path = self._write_video(path, frames, fps)
                if saved_path and self._is_canceled_path(saved_path):
                    self._delete_video_file(saved_path)
                    continue
                if saved_path:
                    self.completed.put((object_id, str(saved_path), True))
                else:
                    self.completed.put((object_id, str(path), False))
            except Exception:
                self.completed.put((object_id, str(path), False))

    # Ghi mảng khung hình thành file video, tự động chuyển codec dự phòng nếu lỗi.
    def _write_video(self, path: Path, frames: list[np.ndarray], fps: float) -> Path | None:
        if len(frames) < config.REPLAY_MIN_FRAMES:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._write_video_file(path, frames, config.REPLAY_CODEC, fps) and self._is_valid_video_file(path):
            return path
        fallback_path = path.with_suffix(".avi")
        if self._write_video_file(fallback_path, frames, config.REPLAY_FALLBACK_CODEC, fps) and self._is_valid_video_file(fallback_path):
            return fallback_path
        return None

    # Thực hiện ghi file video bằng thư viện OpenCV.
    @staticmethod
    def _write_video_file(path: Path, frames: list[np.ndarray], codec: str, fps: float) -> bool:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        if not writer.isOpened():
            return False
        try:
            for frame in frames:
                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                writer.write(frame)
        finally:
            writer.release()
        return True

    # Kiểm tra tính hợp lệ của file video sau khi lưu (kích thước có đủ lớn không).
    @staticmethod
    def _is_valid_video_file(path: Path) -> bool:
        try:
            return path.exists() and path.stat().st_size > config.REPLAY_MIN_VALID_BYTES
        except OSError:
            return False

    # Trả về danh sách các đường dẫn dự phòng (cả đuôi .mp4 và .avi).
    @staticmethod
    def _video_path_candidates(path: Path) -> list[Path]:
        candidates = [path]
        if path.suffix.lower() == ".mp4":
            candidates.append(path.with_suffix(".avi"))
        return candidates

    # Kiểm tra xem video này có nằm trong danh sách đã bị hủy bỏ không.
    def _is_canceled_path(self, path: Path) -> bool:
        return any(str(candidate) in self._canceled_paths for candidate in self._video_path_candidates(path))

    # Xóa file video rác/thừa khỏi đĩa cứng.
    @staticmethod
    def _delete_video_file(path: Path) -> None:
        for candidate in EventReplayBuffer._video_path_candidates(path):
            try:
                if candidate.exists():
                    candidate.unlink()
            except OSError:
                pass


