from __future__ import annotations

import unicodedata

import cv2
import numpy as np

import config
from models import *
from roi_manager import *
from target_manager import *
from report_generator import *
# Trình hỗ trợ vẽ Text có font chuẩn (Pillow) thay vì OpenCV.
# OpenCV vẽ text tiếng Việt bị lỗi font, nên class này dùng thư viện PIL
# để render chữ Unicode rồi convert lại thành ma trận cho OpenCV.
class TextPainter:
    # Khởi tạo đối tượng vẽ chữ (TextPainter) sử dụng font hỗ trợ Unicode.
    def __init__(self) -> None:
        self._pil_available = False
        self._pil_image = None
        self._draw = None
        self._font_cache = {}
        try:
            from PIL import Image, ImageDraw, ImageFont

            self.Image = Image
            self.ImageDraw = ImageDraw
            self.ImageFont = ImageFont
            self._font_path = self._find_font()
            self._pil_available = True
        except Exception:
            self._font_path = None

    # Bắt đầu phiên vẽ: chuyển frame OpenCV sang PIL để vẽ chữ Unicode.
    def begin(self, frame) -> None:
        if not self._pil_available:
            return
        self._pil_image = self.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self._draw = self.ImageDraw.Draw(self._pil_image)

    # Kết thúc phiên vẽ: chuyển ảnh PIL ngược lại thành ma trận OpenCV.
    def end(self, frame) -> None:
        if self._pil_image is None:
            return
        frame[:] = cv2.cvtColor(np.array(self._pil_image), cv2.COLOR_RGB2BGR)
        self._pil_image = None
        self._draw = None

    # Vẽ một dòng chữ lên frame (hỗ trợ tiếng Việt qua PIL, fallback sang OpenCV).
    def text(self, frame, text: str, x: int, y: int, size: int = 16, color=(230, 235, 240), bold: bool = False) -> None:
        if self._draw is not None:
            font = self._font(size, bold)
            self._draw.text((int(x), int(y)), text, font=font, fill=(int(color[2]), int(color[1]), int(color[0])))
            return
        safe = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        cv2.putText(frame, safe, (int(x), int(y + size)), cv2.FONT_HERSHEY_SIMPLEX, max(0.38, size / 32), color, 2 if bold else 1, cv2.LINE_AA)

    # Vẽ hình chữ nhật lên frame (dùng PIL hoặc OpenCV).
    def rect(self, frame, x1: int, y1: int, x2: int, y2: int, color=(100, 115, 130), width: int = 1) -> None:
        if self._draw is not None:
            rgb = (int(color[2]), int(color[1]), int(color[0]))
            self._draw.rectangle((int(x1), int(y1), int(x2), int(y2)), outline=rgb, width=width)
            return
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, width)

    # Lấy font Pillow (có cache) theo kích thước và kiểu đậm.
    def _font(self, size: int, bold: bool = False):
        key = (size, bold)
        if key in self._font_cache:
            return self._font_cache[key]
        if self._font_path:
            font = self.ImageFont.truetype(str(self._font_path), size=size)
        else:
            font = self.ImageFont.load_default()
        self._font_cache[key] = font
        return font

    # Tìm file font TTF trên hệ thống (Windows hoặc Linux).
    @staticmethod
    def _find_font() -> Path | None:
        for path in [
            Path("C:/Windows/Fonts/segoeui.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]:
            if path.exists():
                return path
        return None


import radar_hud

# Chịu trách nhiệm render toàn bộ giao diện OSD (On-Screen Display) đè lên luồng video.
# Bao gồm Bounding Box, Info Panel, Radar, Cảnh báo AI, và Telemetry.
class OverlayRenderer:
    # Khởi tạo renderer với một label (huy hiệu) hiển thị danh sách các model AI.
    def __init__(self, yolo_badge: str = "YOLO") -> None:
        self.text = TextPainter()
        self.yolo_badge = yolo_badge
        self.radar = radar_hud.RadarHUD(canvas_size=200, idle_scale=1.0, alert_scale=1.0, sweep_speed_deg=5.0)
        self.camera_ok = True

    # Hàm render chính chạy mỗi frame. Phân nhánh logic:
    # Nếu show_overlay = True -> Vẽ màn hình Dashboard chi tiết.
    # Nếu show_overlay = False -> Vẽ màn hình Minimal (chỉ hiện BBox cơ bản).
    def render(
        self,
        frame,
        targets: list[TargetState],
        selected: TargetState | None,
        telemetry,
        env,
        events,
        show_overlay: bool,
        fps: float,
        now: float,
    ):
        out = frame.copy()
        if config.ROI_ENABLED and (show_overlay or config.SHOW_ROI_IN_CLEAN_MODE):
            self._draw_roi(out, show_overlay)
        if show_overlay:
            self._draw_dashboard(out, frame, targets, selected, telemetry, env, events, fps)
        self._draw_minimal_targets(out, targets, now)
        if config.ROI_EDIT_MODE:
            self._draw_roi_edit(out)

        if config.SHOW_CROSSHAIR:
            center_x, center_y = out.shape[1] // 2, out.shape[0] // 2
            cv2.line(out, (center_x - 15, center_y), (center_x + 15, center_y), (0, 0, 255), 2)
            cv2.line(out, (center_x, center_y - 15), (center_x, center_y + 15), (0, 0, 255), 2)
            
            if selected:
                cx = (selected.bbox[0] + selected.bbox[2]) // 2
                cy = (selected.bbox[1] + selected.bbox[3]) // 2
                cv2.arrowedLine(out, (center_x, center_y), (cx, cy), (255, 0, 255), 2, tipLength=0.05)

        return out

    # Vẽ chế độ Minimal (chỉ hiện Bounding Box và nhãn cơ bản của các mục tiêu).
    def _draw_minimal_targets(self, frame, targets: list[TargetState], now: float) -> None:
        for target in targets:
            if now - target.last_seen > 0.5:
                continue
            if target.operator_decision == "FALSE_ALARM":
                continue
            if target.final_status == "CONFIRMED_VICTIM":
                color = STATUS_COLORS["CONFIRMED_VICTIM"]
            else:
                color = STATUS_COLORS.get(target.display_status, STATUS_COLORS["NORMAL"])
            x1, y1, x2, y2 = target.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if target.rescue_label == "BINH_THUONG":
                label = f"ID {target.object_id} | {target.class_name} | AI {target.rescue_score} | {self._status_vi(target.display_status)}"
            else:
                label = f"ID {target.object_id} | {target.class_name} -> {target.display_label} | AI {target.rescue_score}"
            if target.operator_decision:
                label += f" | Operator: {target.operator_decision}"
            if target.final_status == "CONFIRMED_VICTIM":
                label += " | Final: ĐÃ XÁC NHẬN"
            label_y = y1 + 20 if y1 < 32 else y1 - 8
            self._label(frame, label, x1, label_y, color)
            side_label = f"AI {target.rescue_score}/100"
            if target.final_status == "CONFIRMED_VICTIM":
                side_label = f"FP {target.final_priority}/100"
            self._label(frame, side_label, x2 + 6, max(4, y1), color)

            if target.kps_xy is not None and target.kps_conf is not None and config.SHOW_SKELETON:
                if target.final_status == "CONFIRMED_VICTIM":
                    bone_color = (220, 70, 220)   # Tím (Màu BGR của Confirmed)
                    joint_color = (255, 255, 255) # Trắng
                elif target.display_status == "DANGER":
                    bone_color = (0, 0, 255)
                    joint_color = (0, 165, 255)
                elif target.display_status == "WARNING":
                    bone_color = (0, 255, 255)
                    joint_color = (0, 255, 0)
                else:
                    bone_color = (0, 255, 0)
                    joint_color = (0, 255, 255)
                    
                for idx, (px, py) in enumerate(target.kps_xy):
                    if target.kps_conf[idx] > 0.4:
                        cv2.circle(frame, (int(px), int(py)), 4, joint_color, -1)
                skeleton_pairs = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
                for p1, p2 in skeleton_pairs:
                    if p1 < len(target.kps_xy) and p2 < len(target.kps_xy) and target.kps_conf[p1] > 0.4 and target.kps_conf[p2] > 0.4:
                        pt1 = (int(target.kps_xy[p1][0]), int(target.kps_xy[p1][1]))
                        pt2 = (int(target.kps_xy[p2][0]), int(target.kps_xy[p2][1]))
                        cv2.line(frame, pt1, pt2, bone_color, 2)
    # Vẽ toàn bộ màn hình Dashboard (giao diện điều khiển đầy đủ thông số).
    def _draw_dashboard(self, frame, source_frame, targets, selected, telemetry, env, events, fps) -> None:
        h, w = frame.shape[:2]
        left_w = int(np.clip(w * 0.24, 290, 360))
        right_w = int(np.clip(w * 0.21, 270, 340))
        bottom_h = int(np.clip(h * 0.38, 310, 360))

        self._panel(frame, 0, 0, left_w, h, 0.66)
        self._panel(frame, w - right_w, 0, right_w, h, 0.66)
        self._panel(frame, left_w, h - bottom_h, max(1, w - left_w - right_w), bottom_h, 0.60)

        self.text.begin(frame)
        self._draw_left(frame, telemetry, env, left_w, h, fps)
        radar_pos = self._draw_right(frame, targets, events, telemetry, w - right_w, right_w, h)
        thumbnail_job = self._draw_bottom(frame, source_frame, selected, telemetry, left_w, h - bottom_h, w - left_w - right_w, bottom_h)
        self.text.end(frame)
        if thumbnail_job is not None:
            image, tx, ty, tw, th = thumbnail_job
            self._thumbnail(frame, image, tx, ty, tw, th)
        if radar_pos is not None:
            self.radar.render(frame, custom_x=radar_pos[0], custom_y=radar_pos[1])

    # Vẽ panel bên trái: Telemetry (nhiệm vụ, pin, kết nối) và Thời tiết.
    def _draw_left(self, frame, telemetry, env, panel_w, h, fps) -> None:
        x, y = 18, 18
        self.text.text(frame, "TRẠM MẶT ĐẤT AI", x, y, 20, (255, 255, 255), True)
        y += 30
        self.text.text(frame, f"YOLO: {self.yolo_badge}", x, y, 15)
        y += 22

        self._section(frame, "TRẠNG THÁI TRẠM CHỦ", x, y)
        y += 28
        for line in [
            f"Pin laptop: {telemetry.laptop_battery_text}",
            f"Sạc: {telemetry.laptop_charging_text}",
            f"Internet: {telemetry.internet_status}",
            f"FPS: {fps:.1f}",
            f"Thời gian: {telemetry.timestamp_text}",
        ]:
            self.text.text(frame, line, x, y, 14)
            y += 20

        y += 10
        self._section(frame, "THỜI TIẾT KHU VỰC BAY", x, y)
        y += 28
        if env.has_data and telemetry.internet_status != "Offline":
            weather_lines = [
                f"Vị trí: {env.location_name}",
                f"Nhiệt độ: {self._fmt_value(env.temperature_c, 1, ' °C')}",
                f"Độ ẩm: {self._fmt_value(env.humidity_percent, 0, ' %')}",
                f"Gió: {self._fmt_value(env.wind_speed_ms, 1, ' m/s')}",
                f"Mưa 1h: {self._fmt_value(env.rain_probability_1h, 0, ' %')} | {self._fmt_value(env.precipitation_1h_mm, 1, ' mm')}",
                f"Mưa 3h: {self._fmt_value(env.rain_probability_3h, 0, ' %')} | {self._fmt_value(env.precipitation_3h_mm, 1, ' mm')}",
                f"Đánh giá: {env.assessment}",
                f"Khuyến nghị: {self._shorten(env.recommendation, 31)}",
                f"Nguồn: {env.source}{' - dữ liệu cũ' if env.is_stale else ''}",
                f"Cập nhật: {env.updated_text}",
            ]
        else:
            weather_lines = [
                "Vị trí: Không xác định",
                "Nhiệt độ: Mất kết nối",
                "Độ ẩm: Mất kết nối",
                "Gió: Mất kết nối",
                "Mưa 1h: Mất kết nối",
                "Mưa 3h: Mất kết nối",
                "Đánh giá: Mất kết nối",
                "Khuyến nghị: Mất kết nối",
                f"Nguồn: {env.source}",
                "Cập nhật: Mất kết nối",
            ]
        for line in weather_lines:
            self.text.text(frame, line, x, y, 13)
            y += 19

        y += 12
        self._section(frame, "THAO TÁC", x, y)
        y += 30
        for line in [
            "TAB: Bật/Tắt overlay",
            "N: Chuyển mục tiêu",
            "B: Bật/Tắt tia ngắm",
            "E: Vẽ ROI",
            "X: Xóa ROI",
            "P: Tắt/Bật xương",
            "O: Bật/Tắt hiển thị ROI",
            "W: Cập nhật thời tiết",
            "R: Tạo báo cáo",
            "A: Lịch sử nhật ký",
            "ESC: Thoát chương trình"
        ]:
            if y > h - 30:
                break
            self.text.text(frame, line, x, y, 15)
            y += 22

    # Vẽ panel bên phải: Ra-đa (Radar) hiển thị vị trí tương đối và Danh sách nhật ký sự kiện.
    def _draw_right(self, frame, targets, events, telemetry, x0, panel_w, h) -> None:
        x, y = x0 + 18, 18
        action_h = 116
        log_lines = 5
        self._section(frame, "DANH SÁCH ƯU TIÊN", x, y)
        y += 30
        priority = sorted([t for t in targets if t.operator_decision != "FALSE_ALARM"], key=TargetManager._target_focus_key, reverse=True)[:4]
        if not priority:
            self.text.text(frame, "Chưa có mục tiêu", x, y, 15)
            y += 24
        for idx, target in enumerate(priority, start=1):
            color = STATUS_COLORS.get(target.final_status if target.final_status in STATUS_COLORS else target.display_status, (235, 235, 235))
            tail = f" | {self._status_vi(target.final_status)}" if target.final_status == "CONFIRMED_VICTIM" else ""
            self.text.text(frame, self._shorten(f"{idx}. ID {target.object_id} | {target.display_label} | {target.rescue_score}/100{tail}", max(30, int((panel_w - 34) / 8))), x, y, 15, color)
            y += 24

        y += 16
        self._section(frame, "RADAR DÒ MỤC TIÊU", x, y)
        y += 28
        action_y = h - action_h - 28
        event_block_h = 28 + (log_lines * 20) + 14
        event_y = action_y - event_block_h

        map_x = x
        map_y = y
        map_w = panel_w - 20
        map_h = event_y - map_y - 10
        
        # Đóng lại khung chữ nhật to bao quanh khu vực Radar
        self.text.rect(frame, map_x, map_y, map_x + map_w, map_y + map_h, (100, 115, 130), 1)
        
        # Thay đổi nội dung Text cho phù hợp với hệ thống AI Radar giả lập, an toàn khi bị hội đồng hỏi
        self.text.text(frame, "Trạng thái: Đang dò tìm", map_x + 10, map_y + 14, 14)
        self.text.text(frame, "Góc nhìn Camera: 165°", map_x + 10, map_y + 36, 14)
        self.text.text(frame, "Mục tiêu bám sát: Con người", map_x + 10, map_y + 58, 14)
        
        # Đồng bộ trạng thái Camera
        sync_status = "Bật" if self.camera_ok else "Tắt"
        self.text.text(frame, f"Đồng bộ tọa độ Camera: {sync_status}", map_x + 10, map_y + 80, 14)
        
        # Cập nhật Radar bán nguyệt (Width = 2 * Height)
        radar_size = min(map_w, map_h * 2)
        self.radar.canvas_size = radar_size
        
        # CHỈ đưa vào radar những mục tiêu hợp lệ (đang hiển thị, không bị đánh dấu False Alarm)
        import time
        current_time = time.time()
        active_targets = [t for t in targets if t.operator_decision != "FALSE_ALARM" and current_time - t.last_seen <= 0.5]
        detections = [t.bbox for t in active_targets]
        
        self.radar.update(detections, (config.CAMERA_WIDTH, config.CAMERA_HEIGHT))
        
        radar_x = map_x + (map_w - radar_size) // 2
        # Đẩy radar nhích lên một chút (trừ đi 15 pixel từ mép dưới)
        radar_y = map_y + map_h - (radar_size // 2) - 15
        
        y = event_y
        
        self._section(frame, "NHẬT KÝ SỰ KIỆN", x, y)
        y += 28
        if not events:
            self.text.text(frame, "Chưa có sự kiện", x, y, 14)
            y += 22
        event_max_chars = max(30, int((panel_w - 34) / 8))
        for event in events[:log_lines]:
            if y > action_y - 12:
                break
            self.text.text(frame, self._shorten(event, event_max_chars), x, y, 13)
            y += 20

        y = max(y + 14, action_y)
        self._section(frame, "HÀNH ĐỘNG", x, y)
        y += 28
        actions = [
            ("C", "Xác nhận nạn nhân"),
            ("F", "Báo nhầm (Bỏ qua)"),
            ("T", "Khóa theo dõi mục tiêu"),
            ("S", "Tập trung ưu tiên cao")
        ]
        for key, label in actions:
            if y > h - 18:
                break
            self.text.text(frame, f"{key}: {label}", x, y, 15)
            y += 22
            
        return radar_x, radar_y

    # Vẽ panel bên dưới: Thông tin chi tiết về mục tiêu đang được chọn (Selected Target) và ảnh crop.
    def _draw_bottom(self, frame, source_frame, selected, telemetry, x0, y0, width, height):
        x, y = x0 + 18, y0 + 14
        crop_w = min(160, max(130, int(width * 0.14)))
        crop_h = min(height - 78, 132)
        crop_x = x0 + width - crop_w - 20
        crop_y = y0 + 48
        info_area_w = max(280, crop_x - x - 28)
        target_max_chars = max(38, int(info_area_w / 8))
        self._section(frame, "MỤC TIÊU ĐANG CHỌN", x, y)
        target_y = y + 28
        if selected is None:
            self.text.text(frame, "Chưa có mục tiêu", x, target_y, 16)
        else:
            status_vi = {
                "NORMAL": "BÌNH THƯỜNG",
                "WARNING": "CẢNH BÁO",
                "DANGER": "NGUY HIỂM",
            }.get(selected.display_status, selected.display_status)
            operator_vi = {
                "CONFIRMED": "ĐÃ XÁC NHẬN",
                "FALSE_ALARM": "BÁO NHẦM",
                "TRACK_MORE": "THEO DÕI THÊM",
            }.get(selected.operator_decision or "", "CHƯA XÁC NHẬN")
            final_vi = {
                "NORMAL": "BÌNH THƯỜNG",
                "WARNING": "CẦN THEO DÕI",
                "DANGER": "NGUY HIỂM",
                "CONFIRMED_VICTIM": "NẠN NHÂN ĐÃ XÁC NHẬN",
                "FALSE_ALARM": "BÁO NHẦM",
            }.get(selected.final_status or selected.display_status, selected.final_status or selected.display_status)
            posture_vi = {
                "standing": "đứng",
                "sitting": "ngồi",
                "normal_posture": "bình thường",
                "upright_or_sitting": "đứng/ngồi",
                "sitting_or_standing": "ngồi/đứng",
                "lying_or_fallen": "nghi nằm/ngã",
                "collapsed_or_crouched": "nghi gục/ngã",
                "unknown": "chưa rõ",
            }.get(selected.posture_status, selected.posture_status)
            motion_vi = {
                "stable": "ổn định",
                "stationary": "ngồi yên",
                "slight_motion": "thay đổi nhẹ",
                "moving": "di chuyển rõ",
                "suspicious_stillness": "ít chuyển động, cần theo dõi",
                "dangerous_motionless": "bất động nguy hiểm",
                "unknown": "chưa rõ",
            }.get(selected.motion_status, selected.motion_status)
            gesture_vi = {
                "none": "không có",
                "one_hand_raised": "giơ 1 tay",
                "two_hands_raised": "giơ 2 tay",
                "two_hands_waving": "giơ 2 tay vẫy rõ",
                "waving": "vẫy tay cầu cứu",
                "distress_signal": "tín hiệu cầu cứu",
            }.get(selected.gesture_status, selected.gesture_status)
            if selected.roi_status == "inside_roi":
                roi_vi = "Có"
            elif selected.roi_status == "outside_roi":
                roi_vi = "Không"
            else:
                roi_vi = "ROI tắt"
            pose_vi = {
                "BINH_THUONG": "bình thường",
                "CAN_CUU_GIUP": "cần cứu giúp",
                "NGUY_HIEM": "nguy hiểm",
            }.get(selected.pose_state, selected.pose_state)
            command_vi = selected.suggested_command
            score_breakdown = selected.score_breakdown or {}
            lines = [
                f"ID: {selected.object_id} | YOLO: {selected.class_name} | Tin cậy: {selected.confidence:.2f}",
                f"Rescue Label: {selected.display_label} | AI Score: {selected.rescue_score}/100 | {status_vi}",
                f"Tư thế: {posture_vi}",
                f"Chuyển động mục tiêu: {motion_vi}",
                f"Cử chỉ: {gesture_vi}",
                "CHI TIẾT ĐIỂM",
                f"YOLO: +{score_breakdown.get('yolo', 0)}  |  ROI: +{score_breakdown.get('roi', 0)}",
                f"Tư thế: +{score_breakdown.get('posture', 0)}",
                f"Chuyển động: +{score_breakdown.get('motion', 0)}  |  Cử chỉ: +{score_breakdown.get('gesture', 0)}",
                f"Tổng: {score_breakdown.get('total', selected.rescue_score)}/100",
                f"Trong ROI: {roi_vi} | Điểm ROI: {selected.roi_score}",
                f"Operator: {operator_vi} | Final Status: {final_vi}",
                f"Lệnh gợi ý: {command_vi}",
            ]
            for line in lines:
                if target_y > y0 + height - 30:
                    break
                self.text.text(frame, self._shorten(line, target_max_chars), x, target_y, 14)
                target_y += 19

        thumbnail_job = None
        self.text.rect(frame, crop_x, crop_y, crop_x + crop_w, crop_y + crop_h, (100, 115, 130), 1)
        self.text.text(frame, "Ảnh crop mục tiêu", crop_x + 8, crop_y + 8, 14)
        if selected is not None and selected.latest_crop is not None:
            thumbnail_job = (selected.latest_crop, crop_x + 8, crop_y + 34, crop_w - 16, crop_h - 44)
        elif selected is not None:
            live_crop = selected.crop_from(source_frame)
            if live_crop is not None:
                selected.latest_crop = live_crop
                thumbnail_job = (live_crop, crop_x + 8, crop_y + 34, crop_w - 16, crop_h - 44)
            else:
                self.text.text(frame, "Chưa có crop", crop_x + 12, crop_y + 46, 14)
        else:
            self.text.text(frame, "Chưa có crop", crop_x + 12, crop_y + 46, 14)

        return thumbnail_job

    # Dịch các nhãn trạng thái từ tiếng Anh sang tiếng Việt để hiển thị.
    @staticmethod
    def _status_vi(status: str) -> str:
        return {
            "TRACKING": "THEO DÕI",
            "SEARCHING": "TÌM KIẾM",
            "DANGER_ALERT": "NGUY HIỂM",
            "NORMAL": "BÌNH THƯỜNG",
            "WARNING": "CẢNH BÁO",
            "DANGER": "NGUY HIỂM",
            "IDLE": "CHỜ",
            "VICTIM_CONFIRMED": "ĐÃ XÁC NHẬN",
            "CONFIRMED_VICTIM": "ĐÃ XÁC NHẬN",
            "FALSE_ALARM": "BÁO NHẦM",
        }.get(status or "", status or "")

    # Định dạng các số thực hiển thị với số chữ số thập phân (kèm hậu tố).
    @staticmethod
    def _fmt_value(value: float | None, digits: int, suffix: str = "") -> str:
        if value is None:
            return "NO_DATA"
        return f"{value:.{digits}f}{suffix}"

    # Rút gọn chuỗi văn bản nếu quá dài (thêm dấu ...).
    @staticmethod
    def _shorten(text: str, max_chars: int) -> str:
        text = str(text).replace("\n", " ").strip()
        if len(text) <= max_chars:
            return text
        return text[: max(0, max_chars - 3)].rstrip() + "..."

    # Vẽ đa giác vùng giám sát ROI lên màn hình camera (màu xanh lá/đỏ tùy trạng thái ROI).
    def _draw_roi(self, frame, show_overlay: bool) -> None:
        if not config.ROI_ENABLED:
            return
        pts = roi_polygon_pixels(frame.shape)
        if len(pts) < 3:
            return
        color = (0, 215, 255)
        thickness = 2 if show_overlay else 1
        cv2.polylines(frame, [np.array(pts, dtype=np.int32)], True, color, thickness)
        if show_overlay:
            x, y = pts[0]
            self._label(frame, config.ROI_LABEL, x, max(8, y - 10), color)

    # Vẽ các điểm đang được click chuột trong chế độ thiết lập ROI (ROI Edit Mode).
    def _draw_roi_edit(self, frame) -> None:
        color = (0, 230, 255)
        self._label(frame, "ROI EDIT | Enter: luu | Q: tat", 18, 32, color)
        pts = list(ROI_EDIT_POINTS)
        if not pts:
            return
        arr = np.array(pts, dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(frame, [arr], False, color, 2)
        if len(pts) >= 3:
            cv2.polylines(frame, [arr], True, color, 1)
        for idx, (x, y) in enumerate(pts, start=1):
            cv2.circle(frame, (int(x), int(y)), 5, color, -1)
            cv2.putText(frame, str(idx), (int(x) + 7, int(y) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Vẽ một hình nền (Panel) có độ trong suốt (Alpha blending).
    @staticmethod
    def _panel(frame, x, y, w, h, alpha) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (10, 14, 18), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (65, 75, 85), 1)

    # Vẽ tiêu đề của một phần (Section) trên panel (với gạch chân).
    def _section(self, frame, title, x, y) -> None:
        self.text.text(frame, title, x, y, 16, (120, 220, 255), True)

    # Hàm tĩnh hỗ trợ vẽ nhãn cơ bản bằng OpenCV putText.
    @staticmethod
    def _label(frame, text, x, y, color) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.48
        thickness = 1
        safe = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
        (tw, th), _ = cv2.getTextSize(safe, font, scale, thickness)
        h, w = frame.shape[:2]
        x = max(0, int(x))
        max_width = max(80, w - x - 12)
        if tw + 8 > max_width:
            ellipsis = "..."
            while len(safe) > 8:
                candidate = safe[:-4].rstrip() + ellipsis
                (cw, _), _ = cv2.getTextSize(candidate, font, scale, thickness)
                if cw + 8 <= max_width:
                    safe = candidate
                    tw = cw
                    break
                safe = safe[:-4].rstrip()
            else:
                safe = safe[:8]
                (tw, th), _ = cv2.getTextSize(safe, font, scale, thickness)
        x = max(0, min(x, w - tw - 8))
        y = max(th + 4, min(int(y), h - th - 8))
        cv2.rectangle(frame, (x, y - th - 6), (x + tw + 8, y + 6), (0, 0, 0), -1)
        cv2.rectangle(frame, (x, y - th - 6), (x + tw + 8, y + 6), color, 1)
        cv2.putText(frame, safe, (x + 4, y), font, scale, (245, 245, 245), thickness, cv2.LINE_AA)

    # Vẽ ảnh thumbnail (ảnh crop của mục tiêu) lên panel thông tin.
    @staticmethod
    def _thumbnail(frame, image, x, y, w, h) -> None:
        ih, iw = image.shape[:2]
        scale = min(w / max(1, iw), h / max(1, ih))
        tw = max(1, int(iw * scale))
        th = max(1, int(ih * scale))
        resized = cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA)
        px = x + (w - tw) // 2
        py = y + (h - th) // 2
        frame[py : py + th, px : px + tw] = resized


