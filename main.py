from __future__ import annotations

import time

import cv2
import numpy as np

import config
from models import *
from utils import *
from roi_manager import *
from weather_telemetry import *
from ui_controller import *
from detector import *
from target_manager import *
from event_logger import *
from report_generator import *
from renderer import *
# Mở kết nối camera theo index trong config.
# Trên Windows dùng DirectShow (CAP_DSHOW) để giảm độ trễ.
# Trả về đối tượng VideoCapture nếu thành công, None nếu thất bại.
def open_camera():
    if config.USE_DSHOW_ON_WINDOWS and isinstance(config.CAMERA_INDEX, int):
        cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        return cap
    return None


# Tạo khung hình giả (placeholder) khi camera chưa sẵn sàng.
# Vẽ thanh màu kiểu SMPTE và dòng chữ 'NO CAMERA SIGNAL' lên nền đen.
def synthetic_frame(width: int, height: int, message: str) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (20, 20, 20)
    colors = [(192, 192, 192), (0, 192, 192), (192, 192, 0), (0, 192, 0), (192, 0, 192), (0, 0, 192), (192, 0, 0), (0, 0, 0)]
    bar_w = width // len(colors)
    for i, c in enumerate(colors):
        cv2.rectangle(frame, (i * bar_w, 0), ((i + 1) * bar_w, height), c, -1)
    cv2.rectangle(frame, (width//2 - 200, height//2 - 40), (width//2 + 200, height//2 + 40), (0, 0, 255), -1)
    cv2.rectangle(frame, (width//2 - 200, height//2 - 40), (width//2 + 200, height//2 + 40), (255, 255, 255), 2)
    msg = "NO CAMERA SIGNAL"
    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    cv2.putText(frame, msg, (width//2 - tw//2, height//2 + th//2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    (tw2, th2), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    cv2.putText(frame, message, (width//2 - tw2//2, height//2 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame



# Hàm chính của chương trình - Vòng lặp xử lý AI Pipeline.
# Quy trình: Khởi tạo 4 model AI + camera → Vòng lặp vô tận:
# Đọc frame → Detect (YOLO+Pose+Posture+Rescue) → Cập nhật target →
# Tính Danger Score → Auto-log nạn nhân → Render giao diện → Xử lý phím.
# Trả về 0 nếu thoát bình thường, 1 nếu lỗi camera.
def main() -> int:
    root = app_root()
    paths = ensure_dirs(root)

    detector = Detector(root)
    manager = TargetManager()
    env_provider = WeatherProvider()
    telemetry_provider = TelemetryProvider()
    logger = EventLogger(paths)
    logger.remember(detector.model_summary())
    load_roi_config(paths, logger)
    replay_buffer = EventReplayBuffer(paths)
    report_generator = ReportGenerator(paths, root)
    overlay = OverlayRenderer(detector.yolo_badge())
    tab = TabToggleController()
    log_window = EventLogWindow(logger)

    cap = open_camera()
    if cap is None:
        print(f"Khong mo duoc camera index {config.CAMERA_INDEX}.")
        logger.remember(f"Không mở được camera index {config.CAMERA_INDEX}.", "WARNING")
        if not config.FALLBACK_SYNTHETIC_FRAME:
            return 1

    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(config.WINDOW_NAME, roi_mouse_callback)
    if config.FULLSCREEN:
        cv2.setWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    last_time = time.time()
    fps = 0.0
    running = True
    show_overlay = False
    frame_index = 0

    try:
        while running:
            frame = None
            ok = False
            if cap is not None:
                ok, frame = cap.read()
            if config.CAMERA_SIMULATE_DISCONNECT:
                ok, frame = False, None
            if not ok or frame is None:
                if not config.FALLBACK_SYNTHETIC_FRAME:
                    break
                frame = synthetic_frame(config.CAMERA_WIDTH, config.CAMERA_HEIGHT, "Camera chua san sang - placeholder")
            else:
                frame = cv2.resize(frame, (config.CAMERA_WIDTH, config.CAMERA_HEIGHT))

            now = time.time()
            frame_index += 1
            dt = max(1e-6, now - last_time)
            last_time = now
            fps = (0.9 * fps) + (0.1 * (1.0 / dt)) if fps > 0 else (1.0 / dt)
            replay_buffer.add_frame(frame, now)

            env = env_provider.read()
            overlay.camera_ok = ok
            if frame_index % max(1, config.DETECTION_EVERY_N_FRAMES) == 0:
                raw_detections = detector.detect(frame, now)
            else:
                raw_detections = []

            for object_id, _path, ok in replay_buffer.get_completed():
                if ok:
                    logger.remember(f"Đã lưu replay video cho ID {object_id}")
                else:
                    logger.remember(f"Replay video lỗi hoặc quá ngắn cho ID {object_id}")

            targets = manager.update(raw_detections, frame, now)
            logger.observe_targets(targets)
            mission_state = mission_state_from_targets(targets)
            telemetry = telemetry_provider.read(fps, mission_state)
            selected = manager.selected_target(now)

            for target in targets:
                if logger.should_auto_log_danger(target, now):
                    logger.save_target_event("AUTO_DANGER", target, frame, telemetry, env, "AUTO", replay_buffer)

            tab.poll()
            show_overlay = tab.is_overlay_visible()
            rendered = overlay.render(
                frame,
                targets,
                selected,
                telemetry,
                env,
                logger.recent_events(),
                show_overlay,
                fps,
                now,
            )
            cv2.imshow(config.WINDOW_NAME, rendered)
            log_window.poll()

            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                running = handle_key(key, selected, manager, logger, frame, telemetry, env, replay_buffer, report_generator, paths, tab, log_window, env_provider)
    finally:
        replay_buffer.stop()
        log_window.close()
        if config.GENERATE_REPORT_ON_EXIT:
            try:
                path = report_generator.generate()
                print(f"[REPORT] {path}")
            except Exception as exc:
                print(f"[REPORT_ERROR] {exc}")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    import os
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002)
    raise SystemExit(main())
