from __future__ import annotations

import math

import config
from models import *
from utils import *
from roi_manager import *
# Quản lý vòng đời và điểm số của tất cả mục tiêu được nhận diện trên màn hình.
# Theo dõi theo object_id từ ByteTrack, tính Danger Score và quản lý trạng thái.
class TargetManager:
    # Khởi tạo trình quản lý mục tiêu (lưu trữ danh sách mục tiêu và trạng thái focus).
    def __init__(self) -> None:
        self.targets: dict[int, TargetState] = {}
        self.next_untracked_id = 10000
        self.focus_target_id: int | None = None
        self.focus_locked = False
        self.focus_manual_until = 0.0

    # Hàm cập nhật vòng lặp chính (chạy theo FPS của camera/AI): nhận raw detection từ detector,
    # mapping với object_id cũ, chạy engine tính Danger Score, và cập nhật nhãn.
    def update(self, detections: list[RawDetection], frame, timestamp: float) -> list[TargetState]:
        polygon = roi_polygon_pixels(frame.shape)
        for det in detections:
            object_id = det.object_id if det.object_id is not None else self.next_untracked_id
            if det.object_id is None:
                self.next_untracked_id += 1

            target = self.targets.get(object_id)
            if target is None:
                target = TargetState(object_id, det.bbox, det.class_name, det.confidence, timestamp, timestamp)
                self.targets[object_id] = target

            analysis = det.analysis or {}
            rescue_match = analysis.get("rescue_match") or {}
            target.bbox = det.bbox
            target.class_name = det.class_name
            raw_posture_class = str(analysis.get("posture_class", "") or "").strip()
            target.posture_class = normalize_posture_class(raw_posture_class) if raw_posture_class else "unknown"
            target.confidence = det.confidence
            target.last_seen = timestamp
            target.pose_state = str(analysis.get("state", "BINH_THUONG"))
            target.gesture_candidate = analysis.get("gesture_candidate", "none")
            target.pose_reasons = list(analysis.get("reasons", []) or [])
            target.detector_source = str(analysis.get("source", "unknown"))
            target.kps_xy = analysis.get("kps_xy")
            target.kps_conf = analysis.get("kps_conf")
            target.pose_flags = {
                "is_lying": bool(analysis.get("is_lying", False)),
                "is_immobile": bool(analysis.get("is_immobile", False)),
                "is_raising_hand": bool(analysis.get("is_raising_hand", False)),
                "is_one_hand_raised": bool(analysis.get("is_one_hand_raised", False)),
                "is_two_hands_raised": bool(analysis.get("is_two_hands_raised", False)),
                "is_waving": bool(analysis.get("is_waving", False)),
                "is_collapsed": bool(analysis.get("is_collapsed", False)),
                "rescue_high_conf": bool(rescue_match.get("is_high_conf", False)),
            }
            target.rescue_victim_evidence = float(analysis.get("rescue_victim_evidence", 0.0))
            target.class_history.append(det.class_name)
            if target.posture_class != "unknown":
                target.posture_class_history.append(target.posture_class)
            target.confidence_history.append(det.confidence)
            target.stable_class = self._stable_class(target)
            if len(target.class_history) >= 3:
                target.class_name = target.stable_class
            target.stable_posture_class = self._stable_posture_class(target)
            left_wrist_sample = analysis.get("left_wrist_sample")
            right_wrist_sample = analysis.get("right_wrist_sample")
            if left_wrist_sample is not None:
                target.left_wrist_history.append(tuple(left_wrist_sample))
            if right_wrist_sample is not None:
                target.right_wrist_history.append(tuple(right_wrist_sample))

            cx, cy = bbox_center(det.bbox)
            target.center_history.append((timestamp, cx, cy))
            status_map = {"standing_person": "standing", "sitting_person": "sitting", "lying_person": "lying_or_fallen"}
            class_posture = status_map.get(target.stable_posture_class, target.stable_posture_class)
            fused_posture = str(analysis.get("posture_status", "unknown") or "unknown")
            posture_confidence = str(analysis.get("posture_confidence", "weak") or "weak")
            if fused_posture in {"standing", "sitting", "lying_or_fallen"}:
                raw_posture_status = fused_posture
            elif class_posture in {"standing", "sitting", "lying_or_fallen"}:
                raw_posture_status = class_posture
            else:
                raw_posture_status = self._analyze_posture(det.bbox, det.confidence)
            target.posture_status = self._resolve_posture_status(target, raw_posture_status, posture_confidence)
            if config.ROI_ENABLED:
                target.roi_status = "inside_roi" if point_in_polygon((cx, cy), polygon) else "outside_roi"
            else:
                target.roi_status = "roi_disabled"
            target.motion_status = self._analyze_motion(target, timestamp)
            self._update_gesture_status(target, str(analysis.get("gesture_candidate", "none") or "none"), timestamp)
            target.suggested_command = self._suggest_command((cx, cy), frame.shape)
            target.latest_crop = target.crop_from(frame)
            self._score_target(target, timestamp)

        self._drop_old_targets(timestamp)
        return list(self.targets.values())

    # Lấy class ổn định của mục tiêu bằng biểu quyết đa số từ 10 frame gần nhất.
    @staticmethod
    def _stable_class(target: TargetState) -> str:
        recent = list(target.class_history)[-10:]
        if not recent:
            return "victim_person" if target.class_name == "victim_person" else "normal_person"
        if target.stable_class == "victim_person":
            return "normal_person" if len(recent) >= 3 and all(item != "victim_person" for item in recent[-3:]) else "victim_person"
        if target.stable_class == "normal_person":
            return "victim_person" if len(recent) >= 3 and all(item == "victim_person" for item in recent[-3:]) else "normal_person"
        if len(recent) >= 3 and all(item == "victim_person" for item in recent[-3:]):
            return "victim_person"
        if len(recent) >= 3 and all(item != "victim_person" for item in recent[-3:]):
            return "normal_person"
        victim_count = sum(1 for item in recent if item == "victim_person")
        normal_count = sum(1 for item in recent if item != "victim_person")
        return "victim_person" if victim_count > normal_count else "normal_person"

    # Lấy tư thế ổn định nhất của mục tiêu (standing/sitting/lying) bằng đếm lịch sử 10 frame.
    @staticmethod
    def _stable_posture_class(target: TargetState) -> str:
        current_posture_class = normalize_posture_class(target.posture_class or "")
        recent = [normalize_posture_class(item) for item in list(target.posture_class_history)[-10:]]
        recent = [item for item in recent if item in {"standing_person", "sitting_person", "lying_person"}]
        if not recent:
            return current_posture_class if current_posture_class == "sitting_person" else "unknown"
        counts = {
            "standing_person": sum(1 for item in recent if item == "standing_person"),
            "sitting_person": sum(1 for item in recent if item == "sitting_person"),
            "lying_person": sum(1 for item in recent if item == "lying_person"),
        }
        highest_count = max(counts.values())
        if highest_count < 2:
            return current_posture_class
        if current_posture_class in counts and counts[current_posture_class] == highest_count:
            return current_posture_class
        return max(counts, key=counts.get)

    # Phân tích tư thế dự phòng bằng tỉ lệ BBox khi Posture model không có kết quả.
    def _analyze_posture(self, bbox, confidence: float) -> str:
        x1, y1, x2, y2 = bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        if confidence < config.POSTURE_LOW_CONFIDENCE or min(width, height) < config.MIN_POSTURE_BBOX_SIZE:
            return "unknown"
        ratio = width / height
        if ratio >= max(config.LYING_RATIO_THRESHOLD, 1.8):
            return "lying_or_fallen"
        if config.POSTURE_MODE == "drone_conservative":
            return "upright_or_sitting"
        elif config.POSTURE_MODE == "detailed":
            vertical_ratio = height / width
            if vertical_ratio >= config.STANDING_RATIO_THRESHOLD:
                return "standing"
            if vertical_ratio >= 0.85:
                return "sitting"
            return "sitting"
        else:
            return "unknown"

    # Xác định tư thế cuối cùng bằng cơ chế xác nhận liên tục (debounce) để tránh nhảy trạng thái.
    @staticmethod
    def _resolve_posture_status(target: TargetState, raw_posture: str, confidence: str = "weak") -> str:
        raw = raw_posture if raw_posture in {"lying_or_fallen", "upright_or_sitting", "standing", "sitting", "unknown"} else "unknown"
        is_half_body = False
        if target.kps_conf is None:
            is_half_body = True
        elif len(target.kps_conf) >= 17:
            if max(target.kps_conf[13:17]) < 0.2:
                is_half_body = True
        if raw in {"standing", "sitting"} and (confidence == "weak" or is_half_body):
            raw = "upright_or_sitting"
            
        if config.POSTURE_MODE == "drone_conservative" and raw in {"standing", "sitting"}:
            raw = "upright_or_sitting"

        target.posture_status_history.append(raw)
        recent = list(target.posture_status_history)[-config.POSTURE_CONFIRM_WINDOW_FRAMES:]
        n = len(recent)

        if config.POSTURE_MODE == "detailed":
            recent_lying = recent[-config.POSTURE_LYING_CONFIRM_FRAMES:]
            if len(recent_lying) >= config.POSTURE_LYING_CONFIRM_FRAMES and all(item == "lying_or_fallen" for item in recent_lying):
                return "lying_or_fallen"

            counts = {}
            for item in recent:
                counts[item] = counts.get(item, 0) + 1
            majority_status = max(counts, key=lambda k: counts[k])
            majority_ratio = counts[majority_status] / max(n, 1)

            if majority_status == "standing" and majority_ratio >= config.POSTURE_MAJORITY_THRESHOLD:
                if target.posture_status == "sitting":
                    if majority_ratio >= (config.POSTURE_MAJORITY_THRESHOLD + 0.10):
                        return "standing"
                else:
                    return "standing"

            if majority_status == "sitting" and majority_ratio >= config.POSTURE_MAJORITY_THRESHOLD:
                if target.posture_status == "standing":
                    if majority_ratio >= (config.POSTURE_MAJORITY_THRESHOLD + 0.10):
                        return "sitting"
                else:
                    return "sitting"
                    
            if majority_status == "upright_or_sitting" and majority_ratio >= config.POSTURE_MAJORITY_THRESHOLD:
                return "upright_or_sitting"

            if target.posture_status in {"standing", "sitting", "lying_or_fallen", "upright_or_sitting"}:
                return target.posture_status

            if n > 0 and counts.get(raw, 0) >= max(2, int(n * 0.4)) and raw in {"standing", "sitting", "upright_or_sitting"}:
                return raw
            return "unknown"

        elif config.POSTURE_MODE == "drone_conservative":
            upright_count = sum(1 for item in recent if item == "upright_or_sitting")
            recent_upright = recent[-config.POSTURE_UPRIGHT_CLEAR_FRAMES:]
            if len(recent_upright) >= config.POSTURE_UPRIGHT_CLEAR_FRAMES and all(item == "upright_or_sitting" for item in recent_upright):
                return "upright_or_sitting"
            recent_lying = recent[-config.POSTURE_LYING_CONFIRM_FRAMES:]
            if len(recent_lying) >= config.POSTURE_LYING_CONFIRM_FRAMES and all(item == "lying_or_fallen" for item in recent_lying):
                return "lying_or_fallen"
            if target.posture_status == "lying_or_fallen":
                return "lying_or_fallen"
            if upright_count > 0:
                return "upright_or_sitting"
            return "unknown"
        else:
            return "unknown"

    # Phân tích trạng thái chuyển động: stable / moving / suspicious_stillness / dangerous_motionless.
    def _analyze_motion(self, target: TargetState, timestamp: float) -> str:
        smooth_history = self._smoothed_center_history(list(target.center_history))
        if len(smooth_history) < 2:
            return "stable"

        motion_window = [item for item in smooth_history if item[0] >= timestamp - config.MOTION_WINDOW_SECONDS]
        if len(motion_window) < 2:
            motion_window = smooth_history[-2:]
        recent_move = self._max_center_displacement(motion_window)

        short_window = [item for item in smooth_history if item[0] >= timestamp - config.STILLNESS_SECONDS]
        short_duration = short_window[-1][0] - short_window[0][0] if len(short_window) >= 2 else 0.0
        short_move = self._max_center_displacement(short_window) if len(short_window) >= 2 else recent_move

        long_window = [item for item in smooth_history if item[0] >= timestamp - config.LONG_IMMOBILE_SECONDS]
        long_duration = long_window[-1][0] - long_window[0][0] if len(long_window) >= 2 else 0.0
        long_move = self._max_center_displacement(long_window) if len(long_window) >= 2 else recent_move

        very_still = long_duration >= config.LONG_IMMOBILE_SECONDS - 0.5 and long_move < max(12, config.CENTER_JITTER_THRESHOLD * 0.5)
        long_still = short_duration >= config.STILLNESS_SECONDS - 0.5 and short_move < config.CENTER_JITTER_THRESHOLD
        if target.pose_flags.get("is_immobile") and recent_move < config.CENTER_JITTER_THRESHOLD:
            long_still = True

        if very_still:
            return "dangerous_motionless"

        if long_still:
            if self._has_dangerous_motionless_context(target):
                return "dangerous_motionless"
            if timestamp - target.first_seen >= config.STILLNESS_SECONDS * 2:
                return "suspicious_stillness"
            if (
                target.posture_status == "unknown"
                or target.pose_state == "NGHI_NGO"
                or (self._is_victim_like_class(target.class_name) and target.confidence >= 0.45)
                or (target.roi_status == "inside_roi" and target.posture_status in {"lying_or_fallen"})
            ):
                return "suspicious_stillness"
            return "stationary"

        if recent_move < config.CENTER_JITTER_THRESHOLD:
            return "stable"
        if recent_move < config.MOVING_THRESHOLD:
            return "slight_motion"
        return "moving"

    # Làm mượt lịch sử tâm bằng trung bình trượt (Moving Average) để lọc giật.
    @staticmethod
    def _smoothed_center_history(history: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
        smoothed: list[tuple[float, float, float]] = []
        for idx, item in enumerate(history):
            window = history[max(0, idx - config.CENTER_SMOOTHING_FRAMES + 1) : idx + 1]
            avg_x = sum(p[1] for p in window) / len(window)
            avg_y = sum(p[2] for p in window) / len(window)
            smoothed.append((item[0], avg_x, avg_y))
        return smoothed

    # Tính độ dịch chuyển lớn nhất của tâm so với điểm đầu tiên trong lịch sử.
    @staticmethod
    def _max_center_displacement(history: list[tuple[float, float, float]]) -> float:
        if len(history) < 2:
            return 0.0
        x0, y0 = history[0][1], history[0][2]
        return max(math.hypot(x - x0, y - y0) for _, x, y in history)

    # Kiểm tra xem tên class có thuộc nhóm nạn nhân (victim, injured...) hay không.
    @staticmethod
    def _is_victim_like_class(class_name: str) -> bool:
        return normalize_rescue_class(class_name) == "victim_person"

    # Cập nhật trạng thái cử chỉ tay (giơ tay, vẫy tay) với cơ chế hold time.
    def _update_gesture_status(self, target: TargetState, raw_candidate: str, timestamp: float) -> None:
        candidate = raw_candidate if raw_candidate in {"one_hand_raised", "two_hands_raised", "two_hands_waving"} else "none"
        left_wave = detect_wrist_waving(target.left_wrist_history, timestamp)
        right_wave = detect_wrist_waving(target.right_wrist_history, timestamp)
        if candidate == "two_hands_raised" and left_wave and right_wave:
            candidate = "two_hands_waving"

        if candidate != "none":
            if target.gesture_candidate != candidate:
                target.gesture_candidate = candidate
                target.gesture_candidate_since = timestamp
            required_hold = config.HAND_RAISE_HOLD_SECONDS
            if timestamp - target.gesture_candidate_since >= required_hold:
                target.gesture_status = candidate
                target.gesture_last_detected_time = timestamp
            return

        target.gesture_candidate = "none"
        target.gesture_candidate_since = 0.0
        if timestamp - target.gesture_last_detected_time > config.GESTURE_HOLD_AFTER_DETECT_SECONDS:
            target.gesture_status = "none"

    # ENGINE TÍNH ĐIỂM DANGER SCORE: Cộng điểm từ 5 nguồn: YOLO, Posture, Motion, Gesture, ROI.
    def _score_target(self, target: TargetState, timestamp: float) -> None:
        current_class = target.class_name
        stable_class = target.stable_class or current_class
        stable_posture_class = normalize_posture_class(target.stable_posture_class or target.posture_class or "")

        victim_like = current_class == "victim_person" or (stable_class == "victim_person" and target.posture_status == "lying_or_fallen")
        yolo_score = 0
        posture_score = 0
        motion_score = 0
        gesture_score = 0
        roi_score = 0
        score_reasons: list[str] = []

        yolo_base = int(target.rescue_victim_evidence * 10)

        pose_flags = target.pose_flags or {}
        pose_boost = 0
        pose_boost_reason = ""

        if target.posture_status == "lying_or_fallen":
            pose_boost = 25
            pose_boost_reason = "Posture: dang nam/nga"
        elif target.gesture_status == "two_hands_waving":
            pose_boost = 25
            pose_boost_reason = "Pose: vay 2 tay"
        elif target.gesture_status == "two_hands_raised":
            pose_boost = 20
            pose_boost_reason = "Pose: gio 2 tay"
        elif pose_flags.get("is_collapsed"):
            pose_boost = 15
            if target.posture_status == "sitting":
                pose_boost_reason = "Pose: ngoi guc / tua dau"
            else:
                pose_boost_reason = "Pose: guc nga / dau cheo than"

        if pose_boost > yolo_base:
            yolo_score = pose_boost
            if yolo_base > 0:
                score_reasons.append(f"YOLO: victim ({yolo_base}d) nhung Pose boost len {yolo_score} ({pose_boost_reason})")
            else:
                score_reasons.append(f"YOLO boost tu Pose/Posture: {pose_boost_reason} +{yolo_score}")
        else:
            yolo_score = yolo_base
            if yolo_score > 0:
                score_reasons.append(f"YOLO: model Rescue tin cay -> +{yolo_score}")

        if target.posture_status == "lying_or_fallen":
            posture_score = 30
            score_reasons.append("Tư thế: nghi nằm/ngã +30")

        if target.motion_status == "dangerous_motionless":
            motion_score = config.IMMOBILE_MOTION_SCORE
            score_reasons.append(f"Chuyển động: bất động lâu +{config.IMMOBILE_MOTION_SCORE}")
        elif target.motion_status == "suspicious_stillness":
            motion_score = 5
            score_reasons.append("Chuyển động: ít chuyển động +5")

        gesture_scores = {
            "one_hand_raised": 15,
            "two_hands_raised": 30,
            "two_hands_waving": 50,
        }
        gesture_reasons = {
            "one_hand_raised": "Cử chỉ: giơ 1 tay",
            "two_hands_raised": "Cử chỉ: giơ 2 tay",
            "two_hands_waving": "Cử chỉ: giơ 2 tay vẫy rõ",
        }
        gesture_score = gesture_scores.get(target.gesture_status, 0)
        if gesture_score:
            reason = gesture_reasons.get(target.gesture_status)
            score_reasons.append(f"{reason or 'Cử chỉ'} +{gesture_score}")
        elif target.pose_state == "CAN_CUU_GIUP":
            score_reasons.append("Pose: CAN_CUU_GIUP, không cộng điểm cử chỉ")

        roi_score, roi_reason = self._roi_score(target)
        target.roi_score = roi_score
        target.roi_reason = roi_reason
        if roi_score:
            score_reasons.append(f"ROI: {roi_reason} +{roi_score}")

        if target.operator_decision == "TRACK_MORE":
            score_reasons.append("Operator: TRACK_MORE +0")
        elif target.operator_decision == "CONFIRMED":
            score_reasons.append("Operator: CONFIRMED, không cộng vào AI Score")
        elif target.operator_decision == "FALSE_ALARM":
            score_reasons.append("Operator: FALSE_ALARM, không cộng vào AI Score")

        classical_score = yolo_score + posture_score + motion_score + gesture_score + roi_score

        if target.posture_status == "lying_or_fallen" and target.gesture_status == "two_hands_waving":
            classical_score = max(classical_score, 75)
            score_reasons.append("Nằm/ngã + giơ 2 tay vẫy rõ -> tối thiểu 75")
        if target.roi_status == "inside_roi" and not self._has_real_warning_signal(target):
            classical_score = min(classical_score, config.ROI_NORMAL_PERSON_SCORE_CAP)

        score = int(clamp(classical_score, 0, 100))
        if not score_reasons or (yolo_score + posture_score + motion_score + gesture_score + roi_score) == 0:
            score_reasons = ["Không có dấu hiệu nguy hiểm rõ ràng"]

        raw_bucket = self._status_bucket(score, target)
        target.raw_rescue_score = int(clamp(classical_score, 0, 100))
        target.rescue_score = score
        if score >= config.DANGER_THRESHOLD:
            target.class_name = "victim_person"
        target.score_breakdown = {
            "yolo": yolo_score,
            "posture": posture_score,
            "motion": motion_score,
            "gesture": gesture_score,
            "roi": roi_score,
            "total": score,
        }
        target.score_reasons = score_reasons
        target.score_history.append(score)
        target.raw_status_history.append(raw_bucket)

        if target.operator_decision == "FALSE_ALARM":
            if timestamp < target.suppress_alert_until:
                target.final_status = "FALSE_ALARM"
                target.final_priority = 0
                self._update_rescue_labels(target)
                return
            else:
                target.operator_decision = None

        display_status = raw_bucket
        analysis_status = display_status
        if target.operator_decision == "TRACK_MORE" and timestamp < target.track_more_until:
            analysis_status = "TRACK_MORE"
        target.display_status = display_status
        target.analysis_status = analysis_status
        self._update_final_status(target)
        self._update_rescue_labels(target)

    # Tính điểm ROI: Cộng điểm cơ sở nếu nằm trong vùng (Point-in-Polygon).
    # Cộng thêm bonus nếu phát hiện đối tượng rơi vào tình trạng nguy hiểm trong vùng (nằm/ngã, bất động...).
    def _roi_score(self, target: TargetState) -> tuple[int, str]:
        if not config.ROI_ENABLED or target.roi_status == "roi_disabled":
            return 0, "ROI tắt"
        if target.roi_status != "inside_roi":
            return 0, "ngoài vùng giám sát"

        score = config.ROI_BASE_SCORE
        reasons = ["trong vùng giám sát"]
        if self._is_victim_like_class(target.class_name):
            score += config.ROI_CONTEXT_BONUS_SCORE
            reasons.append("nghi victim")
        if target.posture_status == "lying_or_fallen":
            score += config.ROI_CONTEXT_BONUS_SCORE
            reasons.append("nghi nằm/ngã")
        if target.motion_status == "dangerous_motionless":
            score += config.ROI_CONTEXT_BONUS_SCORE
            reasons.append("bất động nguy hiểm")
        if target.gesture_status in {"two_hands_raised", "two_hands_waving"}:
            score += config.ROI_CONTEXT_BONUS_SCORE
            reasons.append("cử chỉ cầu cứu")
        return min(score, config.ROI_MAX_SCORE), " + ".join(reasons)

    # Kiểm tra mục tiêu có ngữ cảnh nguy hiểm để nâng cấp bất động thành "dangerous_motionless".
    def _has_dangerous_motionless_context(self, target: TargetState) -> bool:
        return (
            target.posture_status in {"lying_or_fallen"}
            or (self._is_victim_like_class(target.class_name) and target.confidence >= config.RESCUE_HIGH_CONF_THRESHOLD)
            or bool(target.pose_flags.get("is_lying"))
            or bool(target.pose_flags.get("is_collapsed"))
            or target.operator_decision == "CONFIRMED"
            or (target.operator_decision == "TRACK_MORE" and target.display_status in {"WARNING", "DANGER"})
        )

    # Kiểm tra mục tiêu có dấu hiệu cảnh báo thật sự (không phải chỉ do ROI).
    def _has_real_warning_signal(self, target: TargetState) -> bool:
        return (
            self._is_victim_like_class(target.class_name)
            or target.posture_status in {"lying_or_fallen"}
            or target.motion_status == "dangerous_motionless"
            or target.gesture_status in {"one_hand_raised", "two_hands_raised", "two_hands_waving"}
            or target.pose_state in {"CAN_CUU_GIUP", "NGHI_NGO", "NGUY_HIEM"}
            or bool(target.pose_flags.get("is_lying"))
            or bool(target.pose_flags.get("is_collapsed"))
            or bool(target.pose_flags.get("is_waving"))
            or bool(target.pose_flags.get("is_raising_hand"))
            or target.operator_decision in {"TRACK_MORE", "CONFIRMED"}
        )

    # Phân nhóm điểm số thành 3 mức: NORMAL / WARNING / DANGER.
    def _status_bucket(self, score: int, target: TargetState) -> str:
        if score >= config.DANGER_THRESHOLD:
            return "DANGER"
        if score >= config.WARNING_THRESHOLD:
            return "WARNING"
        return "NORMAL"

    # Cập nhật trạng thái cuối cùng dựa trên quyết định của Operator (Confirmed/False Alarm).
    @staticmethod
    def _update_final_status(target: TargetState) -> None:
        if target.operator_decision == "CONFIRMED":
            target.final_status = "CONFIRMED_VICTIM"
            target.final_priority = max(95, target.rescue_score)
        elif target.operator_decision == "FALSE_ALARM":
            target.final_status = "FALSE_ALARM"
            target.final_priority = 0
        else:
            target.final_status = target.display_status
            target.final_priority = target.rescue_score

    # Dán nhãn cảnh báo (BÌNH THƯỜNG / CẦN THEO DÕI / NGHI NẠN NHÂN)
    # tùy thuộc vào Danger Score, trạng thái chuyển động và can thiệp của operator.
    @staticmethod
    def _update_rescue_labels(target: TargetState) -> None:
        if target.operator_decision == "CONFIRMED":
            target.rescue_label = "DA_XAC_NHAN"
            target.display_label = "ĐÃ XÁC NHẬN NẠN NHÂN"
            return
        if target.operator_decision == "FALSE_ALARM":
            target.rescue_label = "BAO_NHAM"
            target.display_label = "BÁO NHẦM"
            return
        if target.posture_status in {"lying_or_fallen"} and target.motion_status == "dangerous_motionless":
            target.rescue_label = "NGUY_HIEM"
            target.display_label = "NGHI NẠN NHÂN"
            return
        if target.display_status == "DANGER":
            target.rescue_label = "NGUY_HIEM"
            target.display_label = "NGHI NẠN NHÂN"
            return
        if target.gesture_status in {"two_hands_raised", "two_hands_waving"} and target.rescue_score >= config.WARNING_THRESHOLD:
            target.rescue_label = "NGHI_NAN_NHAN"
            target.display_label = "NGHI NẠN NHÂN"
            return
        if target.display_status == "WARNING":
            target.rescue_label = "CAN_THEO_DOI"
            target.display_label = "CẦN THEO DÕI"
            return
        if TargetManager._is_victim_like_class(target.class_name):
            target.rescue_label = "CAN_THEO_DOI"
            target.display_label = "CẦN THEO DÕI"
            return
        target.rescue_label = "BINH_THUONG"
        target.display_label = "BÌNH THƯỜNG"


    # Gợi ý hướng xoay Drone cho phi công: tính toán vector lệch giữa
    # tâm mục tiêu và tâm màn hình, từ đó xuất ra lệnh Lệch phải/trái/tiến/lùi.
    @staticmethod
    def _suggest_command(center: tuple[float, float], frame_shape) -> str:
        h, w = frame_shape[:2]
        error_x = center[0] - (w / 2)
        error_y = center[1] - (h / 2)
        if abs(error_x) < w * 0.08 and abs(error_y) < h * 0.08:
            return "Giữ vị trí (đã vào tâm ngắm)"
        if abs(error_x) >= abs(error_y):
            if error_x > 0:
                return f"Lệch phải ({int(abs(error_x))}px) -> Xoay phải"
            else:
                return f"Lệch trái ({int(abs(error_x))}px) -> Xoay trái"
        else:
            if error_y < 0:
                return f"Lệch lên ({int(abs(error_y))}px) -> Tiến lên"
            else:
                return f"Lệch xuống ({int(abs(error_y))}px) -> Lùi lại"

    # Chọn mục tiêu ưu tiên nhất để hiển thị và theo dõi trên giao diện.
    def selected_target(self, timestamp: float) -> TargetState | None:
        candidates = self._focus_candidates(timestamp)
        if not candidates:
            self.focus_target_id = None
            self.focus_locked = False
            return None
        best = max(candidates, key=self._target_focus_key)
        current = self.targets.get(self.focus_target_id) if self.focus_target_id is not None else None
        current_valid = (
            current is not None
            and current.operator_decision != "FALSE_ALARM"
            and timestamp - current.last_seen <= config.DROP_TARGET_SECONDS
        )
        if self.focus_locked:
            if current_valid:
                return current
            self.focus_locked = False
            
        if timestamp < getattr(self, 'focus_manual_until', 0.0):
            if current_valid:
                return current
                
        if not current_valid:
            self.focus_target_id = best.object_id
            return best
        if best.object_id == current.object_id:
            return current
        if self._should_switch_focus(current, best):
            self.focus_target_id = best.object_id
            return best
        return current

    # Chuyển lần lượt từng mục tiêu (phím N) theo thứ tự vòng tròn.
    def cycle_focus(self, timestamp: float) -> TargetState | None:
        candidates = sorted(self._focus_candidates(timestamp), key=lambda t: t.object_id)
        if not candidates:
            self.focus_target_id = None
            self.focus_locked = False
            self.focus_manual_until = 0.0
            return None
        self.focus_locked = False
        self.focus_manual_until = timestamp + 10.0
        current_index = next((idx for idx, target in enumerate(candidates) if target.object_id == self.focus_target_id), -1)
        next_target = candidates[(current_index + 1) % len(candidates)] if current_index >= 0 else candidates[0]
        self.focus_target_id = next_target.object_id
        return next_target

    # Khóa ngay vào mục tiêu có độ ưu tiên cao nhất (phím S).
    def focus_highest_priority(self, timestamp: float, lock: bool = True) -> TargetState | None:
        candidates = self._focus_candidates(timestamp)
        if not candidates:
            self.focus_target_id = None
            self.focus_locked = False
            return None
        target = max(candidates, key=self._target_focus_key)
        self.focus_target_id = target.object_id
        self.focus_locked = lock
        self.focus_manual_until = 0.0
        return target

    # Khóa cứng vào một mục tiêu cụ thể theo object_id.
    def lock_focus(self, object_id: int, timestamp: float) -> TargetState | None:
        target = self.targets.get(object_id)
        if target is None or target.operator_decision == "FALSE_ALARM" or timestamp - target.last_seen > config.DROP_TARGET_SECONDS:
            self.focus_locked = False
            return None
        self.focus_target_id = object_id
        self.focus_locked = True
        self.focus_manual_until = 0.0
        return target

    # Lấy danh sách mục tiêu hợp lệ (chưa bị đánh dấu báo nhầm và chưa biến mất).
    def _focus_candidates(self, timestamp: float) -> list[TargetState]:
        return [
            target
            for target in self.targets.values()
            if target.operator_decision != "FALSE_ALARM" and timestamp - target.last_seen <= config.DROP_TARGET_SECONDS
        ]

    # Kiểm tra có nên chuyển focus sang mục tiêu mới không (so sánh độ ưu tiên).
    @staticmethod
    def _should_switch_focus(current: TargetState, candidate: TargetState) -> bool:
        current_confirmed = current.operator_decision == "CONFIRMED" or current.final_status == "CONFIRMED_VICTIM"
        candidate_confirmed = candidate.operator_decision == "CONFIRMED" or candidate.final_status == "CONFIRMED_VICTIM"
        if candidate_confirmed and not current_confirmed:
            return True
        if current_confirmed and not candidate_confirmed:
            return False
        if candidate.display_status == "DANGER" and current.display_status != "DANGER":
            return True
        if candidate.rescue_score >= current.rescue_score + config.FOCUS_SWITCH_SCORE_MARGIN:
            return True
        return False

    # Trả về tuple dùng để so sánh (sort key) độ ưu tiên khi chọn mục tiêu focus.
    @staticmethod
    def _target_focus_key(target: TargetState) -> tuple:
        status_rank = {"NORMAL": 0, "WARNING": 1, "DANGER": 2}
        confirmed_rank = 1 if target.operator_decision == "CONFIRMED" or target.final_status == "CONFIRMED_VICTIM" else 0
        operator_rank = {
            None: 0,
            "TRACK_MORE": 1,
            "CONFIRMED": 3,
        }.get(target.operator_decision, 0)
        danger_signals = 0
        if target.class_name == "victim_person":
            danger_signals += 3
        if target.posture_status in {"lying_or_fallen"}:
            danger_signals += 3
        if target.motion_status == "dangerous_motionless":
            danger_signals += 2
        if target.gesture_status in {"two_hands_waving", "two_hands_raised"}:
            danger_signals += 2
        elif target.gesture_status == "one_hand_raised":
            danger_signals += 1
        if target.roi_status == "inside_roi":
            danger_signals += 1
        x1, y1, x2, y2 = target.bbox
        bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
        priority_score = max(target.final_priority or 0, target.rescue_score)
        return (
            confirmed_rank,
            status_rank.get(target.display_status, 0),
            priority_score,
            danger_signals,
            operator_rank,
            target.confidence,
            bbox_area,
            -target.object_id,
        )

    # Ghi nhận quyết định của Operator: Xác nhận / Báo nhầm / Theo dõi thêm.
    def apply_operator_decision(self, object_id: int, decision: str, timestamp: float) -> None:
        target = self.targets.get(object_id)
        if target is None:
            return
        target.operator_decision = decision
        if decision == "CONFIRMED":
            target.final_status = "CONFIRMED_VICTIM"
            target.final_priority = max(95, target.rescue_score)
        elif decision == "FALSE_ALARM":
            target.final_status = "FALSE_ALARM"
            target.final_priority = 0
            target.suppress_alert_until = timestamp + 10
        elif decision == "TRACK_MORE":
            target.analysis_status = "TRACK_MORE"
            target.track_more_until = timestamp + 10
        self._update_rescue_labels(target)

    # Xóa một mục tiêu khỏi danh sách theo dõi.
    def remove_target(self, object_id: int) -> None:
        self.targets.pop(object_id, None)
        if self.focus_target_id == object_id:
            self.focus_target_id = None
            self.focus_locked = False

    # Loại bỏ các mục tiêu đã biến mất quá lâu (vượt DROP_TARGET_SECONDS).
    def _drop_old_targets(self, timestamp: float) -> None:
        stale_ids = [tid for tid, target in self.targets.items() if timestamp - target.last_seen > config.DROP_TARGET_SECONDS]
        for tid in stale_ids:
            self.targets.pop(tid, None)
            if self.focus_target_id == tid:
                self.focus_target_id = None
                self.focus_locked = False
