from __future__ import annotations

import threading
import time

import config
from roi_manager import *


# Bộ điều khiển phím TAB dùng để chuyển đổi (toggle) chế độ hiển thị Overlay.
# Nhấn TAB để xem Dashboard chi tiết, nhả TAB (hoặc nhấn lại) để ẩn giao diện.
class TabToggleController:
    # Khởi tạo bộ điều khiển phím TAB (toggle overlay dashboard).
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._overlay_visible = False
        self._last_tab_down = False
        self._last_cv_toggle = 0.0
        self._backend = "opencv"
        self._win_user32 = None
        self._install_global_backend()

    # Kiểm tra phím TAB trên Windows (dùng Win32 API GetAsyncKeyState).
    def poll(self) -> None:
        if self._backend != "win32" or self._win_user32 is None:
            return
        try:
            tab_down = bool(self._win_user32.GetAsyncKeyState(0x09) & 0x8000)
        except Exception:
            self._backend = "opencv"
            return

        if tab_down and not self._last_tab_down:
            self._toggle()
        self._last_tab_down = tab_down

    # Nhận phím TAB từ OpenCV waitKey (dùng khi Win32 không khả dụng).
    def update_from_cv_key(self, key: int) -> None:
        if self._backend != "opencv":
            return
        now = time.time()
        if key == 9 and now - self._last_cv_toggle > 0.35:
            self._toggle()
            self._last_cv_toggle = now

    # Trả về True nếu dashboard overlay đang hiển thị.
    def is_overlay_visible(self) -> bool:
        with self._lock:
            return self._overlay_visible

    # Đảo trạng thái hiển thị overlay (toggle).
    def _toggle(self) -> None:
        with self._lock:
            self._overlay_visible = not self._overlay_visible

    # Thử cài đặt backend Win32 (ưu tiên) hoặc fallback về OpenCV.
    def _install_global_backend(self) -> None:
        try:
            import ctypes

            self._win_user32 = ctypes.windll.user32
            self._backend = "win32"
            return
        except Exception:
            self._win_user32 = None
            self._backend = "opencv"


# Bộ lắng nghe sự kiện phím tắt (Keyboard Shortcut Handler).
# Điều hướng thao tác tay cầm và phím nóng: Lưu thủ công, Xác nhận/Báo nhầm/Theo dõi thêm,
# Bật/tắt ROI, Thoát chương trình (Q).
def handle_key(key: int, selected, manager, logger, frame, telemetry, env, replay_buffer, report_generator, paths, tab, log_window, env_provider) -> bool:
    if key == 9:
        tab.update_from_cv_key(key)
        return True
    if key in (ord("a"), ord("A")):
        log_window.open()
        return True
    if key in (ord("w"), ord("W")):
        if telemetry.internet_status == "Offline":
            logger.remember("Lỗi: Mất kết nối mạng, không thể làm mới thời tiết.", "WARNING")
        else:
            logger.remember("Đang yêu cầu làm mới dữ liệu thời tiết (tải ngầm)...", "SYSTEM")
            env_provider.read(force=True)
        return True
    if key == 27:
        return False
    if key in (ord("q"), ord("Q")):
        if config.ROI_EDIT_MODE:
            cancel_roi_edit(logger)
        return True
    if key < 0:
        return True
    if key in (13, 10):
        if config.ROI_EDIT_MODE:
            save_roi_edit(paths, logger)
            refresh_target_roi_status(manager)
        return True

    key_char = chr(key).lower() if 0 <= key <= 255 else ""
    if key_char == "r":
        try:
            path = report_generator.generate()
            logger.remember(f"Đã tạo báo cáo HTML: {path.name}", "ACTION")
        except Exception as exc:
            logger.remember(f"Lỗi tạo báo cáo HTML: {exc}", "WARNING")
        return True
    if key_char == "o":
        toggle_roi(paths, logger)
        refresh_target_roi_status(manager)
        return True
    if key_char == "e":
        start_roi_edit(logger)
        return True
    if key_char == "x":
        clear_roi(paths, logger)
        refresh_target_roi_status(manager)
        return True
    if key_char == "v":
        config.CAMERA_SIMULATE_DISCONNECT = not config.CAMERA_SIMULATE_DISCONNECT
        state = "Ngắt" if config.CAMERA_SIMULATE_DISCONNECT else "Khôi phục"
        logger.remember(f"{state} kết nối Camera (Giả lập).", "ACTION")
        return True
    if key_char == "p":

        config.SHOW_SKELETON = not config.SHOW_SKELETON
        logger.remember(f"{'Bật' if config.SHOW_SKELETON else 'Tắt'} khung xương 17 điểm.", "ACTION")
        return True

    now = time.time()
    if key_char == "n":
        target = manager.cycle_focus(now)
        if target is None:
            logger.remember("Không có mục tiêu để chuyển.", "WARNING")
        elif manager.focus_locked:
            logger.remember(f"Chuyển khóa mục tiêu sang ID {target.object_id}.", "ACTION", dedupe_seconds=0)
        else:
            logger.remember(f"Chuyển mục tiêu sang ID {target.object_id}.", "ACTION", dedupe_seconds=0)
        return True
    if key_char == "b":

        config.SHOW_CROSSHAIR = not config.SHOW_CROSSHAIR
        logger.remember(f"{'Bật' if config.SHOW_CROSSHAIR else 'Tắt'} tia ngắm trung tâm.", "ACTION")
        return True
    if key_char == "s":
        target = manager.focus_highest_priority(now, lock=True)
        if target is None:
            logger.remember("Không có mục tiêu ưu tiên để khóa.", "WARNING")
        else:
            logger.remember(f"Đã khóa mục tiêu ưu tiên nhất: ID {target.object_id}.", "ACTION", dedupe_seconds=0)
        return True

    if selected is None:
        if key_char in {"c", "f", "t"}:
            logger.remember("Không có mục tiêu đang chọn.", "WARNING")
        return True

    if key_char == "c":
        manager.apply_operator_decision(selected.object_id, "CONFIRMED", now)
        logger.save_target_event("OPERATOR_CONFIRMED", selected, frame, telemetry, env, "CONFIRMED", replay_buffer)
        logger.remember(f"Người vận hành xác nhận ID {selected.object_id} là nạn nhân.", "ACTION", dedupe_seconds=0)
    elif key_char == "f":
        manager.apply_operator_decision(selected.object_id, "FALSE_ALARM", now)
        logger.remove_saved_media_for_target(selected.object_id, replay_buffer)
        logger.save_target_event("OPERATOR_FALSE_ALARM", selected, frame, telemetry, env, "FALSE_ALARM", replay_buffer)
        logger.remember(f"Người vận hành đánh dấu ID {selected.object_id} là báo nhầm.", "ACTION", dedupe_seconds=0)
    elif key_char == "t":
        manager.apply_operator_decision(selected.object_id, "TRACK_MORE", now)
        manager.lock_focus(selected.object_id, now)
        logger.save_target_event("OPERATOR_TRACK_MORE", selected, frame, telemetry, env, "TRACK_MORE", replay_buffer)
        logger.remember(f"Khóa mục tiêu ID {selected.object_id} để tiếp tục theo dõi.", "ACTION", dedupe_seconds=0)
    return True
