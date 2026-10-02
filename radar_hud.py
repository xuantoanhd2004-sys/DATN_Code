"""
radar_hud.py
------------
Overlay HUD kieu "man hinh radar" (giong anh tham khao) de gan len video
cua drone dang chay computer vision.

Hanh vi:
- Radar luon quet lien tuc (tia sang xoay + duoi mo dan).
- Khi KHONG phat hien nguoi  -> radar o trang thai "idle": nho, mo, nam lui
  ve goc man hinh (background).
- Khi CO phat hien nguoi     -> radar chuyen sang trang thai "alert": phong to,
  sang ro, vien nhap nhay, hien "len truoc" (foreground) va ve cham (blip)
  cho tung nguoi duoc phat hien, tai vi tri (goc phuong vi + khoang cach uoc
  luong) tinh tu bounding box.
- Chuyen trang thai duoc lam muot (lerp) de khong bi giat frame.

Cach dung nhanh:
    from radar_hud import RadarHUD
    radar = RadarHUD()
    ...
    radar.update(person_boxes_xyxy, (frame_w, frame_h))
    frame = radar.render(frame)
"""

import math
import time

import cv2
import numpy as np


def blend_overlay(base, overlay, x, y, alpha=1.0):
    """
    Cong (additive) `overlay` (BGR, nen den = trong suot) len `base` tai
    toa do (x, y) trong `base`. Tu dong cat neu tran bien. Tra ve `base`.
    """
    h, w = overlay.shape[:2]
    H, W = base.shape[:2]

    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return base

    ox0, oy0 = x0 - x, y0 - y
    ox1, oy1 = ox0 + (x1 - x0), oy0 + (y1 - y0)

    roi = base[y0:y1, x0:x1].astype(np.float32)
    ov = overlay[oy0:oy1, ox0:ox1].astype(np.float32) * alpha
    base[y0:y1, x0:x1] = np.clip(roi + ov, 0, 255).astype(np.uint8)
    return base


class RadarHUD:
    def __init__(
        self,
        canvas_size=300,          # do phan giai noi bo cua radar (px)
        margin=20,                # khoang cach tu canh frame (px)
        corner="top-right",       # "top-right" | "top-left" | "bottom-right" | "bottom-left"
        idle_scale=0.55,          # kich thuoc khi khong co nguoi
        idle_alpha=0.45,          # do mo khi khong co nguoi
        alert_scale=1.0,          # kich thuoc khi co nguoi (phong to = "len truoc")
        alert_alpha=1.0,          # do mo khi co nguoi (net = "len truoc")
        sweep_speed_deg=4.0,      # toc do quet, do/frame
        max_bearing_deg=82.5,     # goc quet ngang toi da ung voi FOV 165 do cua camera
        trail_span_deg=55,        # do dai duoi mo phia sau tia quet
        color=(80, 255, 120),     # mau xanh la (BGR)
        transition_speed=0.15,    # toc do lam muot khi doi trang thai (0-1)
    ):
        self.canvas_size = canvas_size
        self.margin = margin
        self.corner = corner
        self.idle_scale = idle_scale
        self.idle_alpha = idle_alpha
        self.alert_scale = alert_scale
        self.alert_alpha = alert_alpha
        self.sweep_speed = sweep_speed_deg
        self.max_bearing = max_bearing_deg
        self.trail_span = trail_span_deg
        self.color = color
        self.smooth = transition_speed

        self.sweep_angle = 0.0
        self.cur_scale = idle_scale
        self.cur_alpha = idle_alpha
        self._blips = []
        self._has_person = False
        self._pulse_t = 0.0
        self.sweep_dir = 1

    # ------------------------------------------------------------------ #
    # API chinh
    # ------------------------------------------------------------------ #
    def update(self, detections, frame_size):
        """
        detections : list[(x1, y1, x2, y2)] toa do pixel bbox nguoi TREN FRAME GOC
        frame_size : (W, H) cua frame goc
        Goi ham nay MOI FRAME, truoc render().
        """
        self.sweep_angle += self.sweep_speed * self.sweep_dir
        if self.sweep_angle >= self.max_bearing:
            self.sweep_angle = self.max_bearing
            self.sweep_dir = -1
        elif self.sweep_angle <= -self.max_bearing:
            self.sweep_angle = -self.max_bearing
            self.sweep_dir = 1
            
        self._pulse_t += 0.12

        has_person = len(detections) > 0
        self._has_person = has_person

        target_scale = self.alert_scale if has_person else self.idle_scale
        target_alpha = self.alert_alpha if has_person else self.idle_alpha
        self.cur_scale += (target_scale - self.cur_scale) * self.smooth
        self.cur_alpha += (target_alpha - self.cur_alpha) * self.smooth

        W, H = frame_size
        W_canvas = self.canvas_size
        H_canvas = W_canvas // 2
        R = H_canvas - 4
        center_x = W_canvas / 2
        center_y = H_canvas - 2

        blips = []
        for (x1, y1, x2, y2) in detections:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            
            # Ánh xạ theo FOV thực tế của Camera (max_bearing = 82.5 độ)
            area_ratio = max((x2 - x1), 1) * max((y2 - y1), 1) / float(W * H)

            # cx càng xa tâm màn hình -> góc bearing càng lớn
            bearing_deg = ((cx - W / 2.0) / (W / 2.0)) * self.max_bearing
            
            # Diện tích càng lớn (càng gần) -> bán kính càng nhỏ (gần tâm radar)
            dist_norm = 1.0 - min(area_ratio / 0.25, 1.0)
            dist_norm = max(dist_norm, 0.08)

            rad = math.radians(bearing_deg)
            # Trong radar, 0 độ là hướng thẳng đứng lên (phía trước)
            bx = center_x + R * dist_norm * math.sin(rad)
            by = center_y - R * dist_norm * math.cos(rad)
            
            blips.append((bx, by))
        self._blips = blips

    def render(self, frame, custom_x=None, custom_y=None):
        """Ve HUD len `frame` (BGR ndarray, sua tai cho) va tra ve frame."""
        canvas = self._draw_canvas()
        
        H_c, W_c = canvas.shape[:2]
        size_w = max(int(W_c * self.cur_scale), 10)
        size_h = max(int(H_c * self.cur_scale), 10)
        canvas_rs = cv2.resize(canvas, (size_w, size_h), interpolation=cv2.INTER_AREA)

        H, W = frame.shape[:2]
        if custom_x is not None and custom_y is not None:
            x, y = int(custom_x), int(custom_y)
        else:
            if self.corner == "top-right":
                x, y = W - size_w - self.margin, self.margin
            elif self.corner == "top-left":
                x, y = self.margin, self.margin
            elif self.corner == "bottom-right":
                x, y = W - size_w - self.margin, H - size_h - self.margin
            else:
                x, y = self.margin, H - size_h - self.margin

        blend_overlay(frame, canvas_rs, x, y, alpha=self.cur_alpha)
        return frame

    # ------------------------------------------------------------------ #
    # Ve noi dung radar (rings, tick, sweep, blip) tren canvas rieng
    # ------------------------------------------------------------------ #
    def _draw_canvas(self):
        W_canvas = self.canvas_size
        H_canvas = W_canvas // 2
        c = np.zeros((H_canvas, W_canvas, 3), dtype=np.uint8)
        
        # Radar bán nguyệt: Tâm nằm ở cạnh dưới
        center = (W_canvas // 2, H_canvas - 2)
        R = H_canvas - 4

        col_dim = tuple(int(v * 0.35) for v in self.color)
        col_mid = tuple(int(v * 0.6) for v in self.color)
        col_bright = self.color

        # Vẽ 4 vòng cung
        for i in range(1, 4):
            cv2.circle(c, center, int(R * i / 3), col_dim, 1, cv2.LINE_AA)
        cv2.circle(c, center, R, col_mid, 2, cv2.LINE_AA)

        # Trục ngang và dọc
        cv2.line(c, (center[0] - R, center[1]), (center[0] + R, center[1]), col_dim, 1, cv2.LINE_AA)
        cv2.line(c, (center[0], center[1] - R), (center[0], center[1]), col_dim, 1, cv2.LINE_AA)

        # Tick marks chỉ vẽ cho nửa trên (-90 đến +90 độ)
        for deg in range(-90, 91, 6):
            a = math.radians(deg)
            r1 = R
            r2 = R - (9 if deg % 30 == 0 else 5)
            p1 = (int(center[0] + r1 * math.sin(a)), int(center[1] - r1 * math.cos(a)))
            p2 = (int(center[0] + r2 * math.sin(a)), int(center[1] - r2 * math.cos(a)))
            cv2.line(c, p1, p2, col_mid, 1, cv2.LINE_AA)

        # duoi mo dan phia sau tia quet
        steps = 40
        for i in range(steps):
            frac = i / steps
            deg = self.sweep_angle - self.sweep_dir * frac * self.trail_span
            a_fade = 0.5 * (1 - frac) ** 2  # Giảm độ sáng của đuôi quét xuống một nửa
            col = tuple(int(v * a_fade) for v in col_bright)
            a0, a1 = math.radians(deg - 0.9), math.radians(deg + 0.9)
            pts = [center]
            for ang in (a0, a1):
                pts.append((int(center[0] + R * math.sin(ang)), int(center[1] - R * math.cos(ang))))
            cv2.fillPoly(c, [np.array(pts, dtype=np.int32)], col, cv2.LINE_AA)

        a = math.radians(self.sweep_angle)
        p_edge = (int(center[0] + R * math.sin(a)), int(center[1] - R * math.cos(a)))
        cv2.line(c, center, p_edge, col_bright, 2, cv2.LINE_AA)
        cv2.circle(c, center, 3, col_bright, -1, cv2.LINE_AA)

        for (bx, by) in self._blips:
            p = (int(bx), int(by))
            pulse = 1.0 + 0.25 * math.sin(self._pulse_t * 3)
            cv2.circle(c, p, int(8 * pulse), col_bright, 1, cv2.LINE_AA)
            cv2.circle(c, p, 3, col_bright, -1, cv2.LINE_AA)

        if self._has_person:
            # Vien nhap nhay bo di vi da bo vien chu nhat
            pass

        return c
