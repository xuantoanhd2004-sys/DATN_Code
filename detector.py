from __future__ import annotations

import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

import config
from models import *
from utils import *
# Lõi AI Pipeline - Quản lý 4 mô hình nhận diện:
# Person (YOLOv8s), Pose (YOLOv8s-pose), Posture (tự train), Rescue (tự train).
# Chạy cascade detect: Nhận diện người → Cắt crop → Pose+Posture+Rescue → Dung hợp.
class Detector:
    # Khởi tạo và load tất cả model AI từ thư mục Model/.
    def __init__(self, root: Path) -> None:
        self.root = root
        self.model_person = None
        self.model_pose = None
        self.model_posture = None
        self.model_rescue = None
        self.person_available = False
        self.pose_available = False
        self.posture_available = False
        self.rescue_available = False
        self.rescue_model_version = 0
        self.rescue_model_name = "NONE"
        self.movement_history = defaultdict(deque)
        self.left_wrist_history = defaultdict(lambda: deque(maxlen=120))
        self.right_wrist_history = defaultdict(lambda: deque(maxlen=120))
        self._load_models()

    # Trả về chuỗi tóm tắt các model đã load thành công (để hiển thị log khởi động).
    def model_summary(self) -> str:
        s = "MODEL:"
        if self.person_available: s += " Person"
        if self.pose_available: s += " Pose"
        if self.posture_available: s += " + Posture"
        if self.rescue_available: s += " + Rescue"
        if not (self.pose_available or self.posture_available or self.rescue_available): return "MODEL: chua load duoc YOLO"
        return s

    # Trả về nhãn hiển thị trên giao diện (ví dụ: 'Person + Pose + Posture + Rescue').
    def yolo_badge(self) -> str:
        parts = []
        if self.person_available: parts.append("Person")
        if self.pose_available: parts.append("Pose")
        if self.posture_available: parts.append("Posture")
        if self.rescue_available: parts.append("Rescue")
        if not parts: return "Not loaded"
        return " + ".join(parts)

    # Nạp 4 model AI từ đĩa: Person, Pose (Ultralytics), Posture và Rescue (tự train).
    # Nếu file model không tồn tại, hàm sẽ bỏ qua và đánh dấu _available = False.
    def _load_models(self) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            print(f"[CANH BAO] Chua co ultralytics YOLO: {exc}")
            return

        try:
            self.model_person = YOLO(config.PERSON_MODEL_PATH)
            self.person_available = True
            print(f"[OK] Person model: {config.PERSON_MODEL_PATH}")
        except Exception as exc:
            print(f"[CANH BAO] Khong load duoc person model: {exc}")
            self.model_person = None

        try:
            self.model_pose = YOLO(config.POSE_MODEL_PATH)
            self.pose_available = True
            print(f"[OK] Pose model: {config.POSE_MODEL_PATH}")
        except Exception as exc:
            print(f"[CANH BAO] Khong load duoc pose model: {exc}")
            self.model_pose = None

        posture_path = self._find_model(config.POSTURE_MODEL_CANDIDATES)
        if posture_path is not None:
            try:
                self.model_posture = YOLO(str(posture_path))
                nc_posture = getattr(self.model_posture.model, "nc", "unknown")
                self.posture_available = True
                print(f"[OK] Posture model: {posture_path.name}, classes={nc_posture}")
            except Exception as exc:
                print(f"[CANH BAO] Khong load duoc posture model: {exc}")
                self.model_posture = None
        else:
            print("[CANH BAO] Khong tim thay mo hinh Posture (theo config).")

        rescue_path = self._find_model(config.RESCUE_MODEL_CANDIDATES)
        if rescue_path is None:
            print("[CANH BAO] Khong tim thay mo hinh Rescue (theo config). Van chay Pose + Posture neu co.")
        else:
            try:
                self.model_rescue = YOLO(str(rescue_path))
                nc = getattr(self.model_rescue.model, "nc", 2)
                self.rescue_model_name = rescue_path.name
                self.rescue_available = True
                print(f"[OK] Rescue model: {rescue_path.name}, classes={nc}")
            except Exception as exc:
                print(f"[CANH BAO] Khong load duoc Rescue model: {exc}")
                self.model_rescue = None

    # Tìm file model (.pt) theo danh sách ưu tiên, trả về Path đầu tiên tồn tại.
    def _find_model(self, candidates) -> Path | None:
        for candidate in candidates:
            path = Path(candidate)
            checks = [self.root / path, self.root.parent / path, self.root / "models" / path.name, path]
            for item in checks:
                if item.exists():
                    return item
        return None

    # Hàm phát hiện chính - Gọi cascade pipeline hoặc simulate.
    # Trả về danh sách RawDetection (bbox, ID, class, confidence, pose, posture, rescue).
    def detect(self, frame, timestamp: float) -> list[RawDetection]:
        self._cleanup_stale_ids(timestamp)
        if self.person_available and self.model_person is not None:
            return self._detect_cascade_pipeline(frame, timestamp)
        if config.SIMULATE_WHEN_MODEL_MISSING:
            return self._simulate(frame, timestamp)
        return []

    # Dọn sạch lịch sử chuyển động của các ID đã biến mất quá lâu.
    def _cleanup_stale_ids(self, current_time: float) -> None:
        if not hasattr(self, '_last_cleanup_time'):
            self._last_cleanup_time = current_time
        if current_time - getattr(self, '_last_cleanup_time', 0) < 5.0:
            return
        self._last_cleanup_time = current_time
        stale_ids = []
        for obj_id, history in self.movement_history.items():
            if not history or (current_time - history[-1][0] > 30.0):
                stale_ids.append(obj_id)
        for obj_id in stale_ids:
            self.movement_history.pop(obj_id, None)
            self.left_wrist_history.pop(obj_id, None)
            self.right_wrist_history.pop(obj_id, None)

    # Pipeline chính: YOLOv8s tìm người (Bounding Box + ByteTrack ID) →
    # Cắt crop từng người → Chạy song song 3 model: Pose (17 keypoints),
    # Posture (Đứng/Ngồi/Nằm), Rescue (Bình thường/Nạn nhân) →
    # Dung hợp kết quả chéo (Late Fusion) → Trả về RawDetection.
    def _detect_cascade_pipeline(self, frame, timestamp: float) -> list[RawDetection]:
        if not self.person_available or self.model_person is None:
            return []

        h, w = frame.shape[:2]
        # 1. Detect humans on full frame
        try:
            results = self.model_person.track(
                source=frame,
                classes=[0],
                conf=0.6,  # Tăng từ 0.4 lên 0.6 để lọc bớt nhiễu (ghế, đồ vật)
                verbose=False,
                persist=True,
                tracker=config.TRACKER_CONFIG,
            )
        except Exception:
            return []
            
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        boxes = results[0].boxes
        xyxy_all = boxes.xyxy.cpu().numpy().astype(int)
        confs_all = boxes.conf.cpu().numpy()
        ids_all = boxes.id.cpu().numpy().astype(int).tolist() if boxes.id is not None else [None] * len(xyxy_all)

        posture_boxes_full = []
        if self.posture_available and self.model_posture is not None:
            try:
                pos_res = self.model_posture.predict(source=frame, conf=0.6, verbose=False)
                if pos_res and pos_res[0].boxes is not None and len(pos_res[0].boxes) > 0:
                    pboxes = pos_res[0].boxes
                    p_xyxy = pboxes.xyxy.cpu().numpy().astype(int)
                    p_cls = pboxes.cls.cpu().numpy()
                    p_conf = pboxes.conf.cpu().numpy()
                    p_names = getattr(pos_res[0], "names", {})
                    for i in range(len(p_xyxy)):
                        posture_boxes_full.append({
                            "box": p_xyxy[i],
                            "cls_id": int(p_cls[i]),
                            "conf": float(p_conf[i]),
                            "names": p_names
                        })
            except Exception:
                pass

        rescue_boxes_full = []
        if self.rescue_available and self.model_rescue is not None:
            try:
                res_res = self.model_rescue.predict(source=frame, conf=0.6, verbose=False)
                if res_res and res_res[0].boxes is not None and len(res_res[0].boxes) > 0:
                    rboxes = res_res[0].boxes
                    r_xyxy = rboxes.xyxy.cpu().numpy().astype(int)
                    r_cls = rboxes.cls.cpu().numpy()
                    r_conf = rboxes.conf.cpu().numpy()
                    r_names = getattr(res_res[0], "names", {})
                    for i in range(len(r_xyxy)):
                        rescue_boxes_full.append({
                            "box": r_xyxy[i],
                            "cls_id": int(r_cls[i]),
                            "conf": float(r_conf[i]),
                            "names": r_names
                        })
            except Exception:
                pass

        detections: list[RawDetection] = []
        for index, (box, conf) in enumerate(zip(xyxy_all, confs_all)):
            x1, y1, x2, y2 = box
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            
            person_box = (x1, y1, x2, y2)
            box_w = x2 - x1
            box_h = y2 - y1
            
            # Tăng kích thước lọc từ 20x40 lên 40x80 (loại bỏ các mảng nhiễu quá nhỏ bé như góc ghế)
            if box_w < 40 or box_h < 80:
                continue

            object_id = int(ids_all[index]) if ids_all[index] is not None else None
            
            crop_frame = frame[y1:y2, x1:x2]
            if crop_frame.size == 0:
                continue

            # Crop image with padding
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x)
            cy2 = min(h, y2 + pad_y)
            
            crop_frame = frame[cy1:cy2, cx1:cx2]
            if crop_frame.size == 0:
                continue

            # 2. Run Pose on crop
            kps_xy = None
            kps_conf = None
            valid_kps = 0
            if self.pose_available and self.model_pose is not None:
                try:
                    p_res = self.model_pose.predict(source=crop_frame, conf=0.3, verbose=False)
                    if p_res and p_res[0].keypoints is not None and len(p_res[0].keypoints) > 0:
                        # Translate keypoints back to full frame
                        kps_xy = p_res[0].keypoints.xy.cpu().numpy()[0]
                        kps_conf = p_res[0].keypoints.conf.cpu().numpy()[0] if p_res[0].keypoints.conf is not None else None
                        # add offset cx1, cy1
                        for kp in kps_xy:
                            kp[0] += cx1
                            kp[1] += cy1
                except Exception:
                    pass

                # Màng lọc bóng ma đã được dời xuống phía dưới sau phần IoU matching
                valid_kps = int(np.sum(kps_conf > 0.25)) if kps_conf is not None else 0

            # 3. Run Posture on full frame + IoU match
            best_posture_iou = 0.0
            best_posture_match = None
            for pb in posture_boxes_full:
                iou = iou_overlap(person_box, pb["box"])
                if iou > best_posture_iou and iou > 0.3:
                    best_posture_iou = iou
                    best_posture_match = pb
            
            posture_match = None
            if best_posture_match:
                custom_status = normalize_posture_class(best_posture_match["names"].get(best_posture_match["cls_id"], "unknown"))
                if custom_status != "unknown":
                    posture_match = {
                        "class_name": custom_status,
                        "confidence": best_posture_match["conf"],
                        "bbox": person_box,
                        "object_id": object_id
                    }

            # 4. Run Rescue on full frame + IoU match
            best_rescue_iou = 0.0
            best_rescue_match = None
            for rb in rescue_boxes_full:
                iou = iou_overlap(person_box, rb["box"])
                if iou > best_rescue_iou and iou > 0.3:
                    best_rescue_iou = iou
                    best_rescue_match = rb
            
            rescue_match = None
            if best_rescue_match:
                raw_name = str(best_rescue_match["names"].get(best_rescue_match["cls_id"], best_rescue_match["cls_id"])).strip().lower()
                cname = normalize_rescue_class(raw_name, best_rescue_match["cls_id"])
                rescue_match = {
                    "class_name": cname,
                    "confidence": best_rescue_match["conf"],
                    "bbox": person_box
                }

            # FUSION
            pose = self._analyze_person_state(person_box, kps_xy, kps_conf, object_id, timestamp)
            pose["kps_xy"] = kps_xy
            pose["kps_conf"] = kps_conf
            pose["bbox"] = person_box  
            pose["confidence"] = float(conf)
            posture_status_dict = self._fuse_posture(posture_match, pose)
            pose.update(posture_status_dict)
            if posture_match is not None:
                pose["posture_match"] = posture_match

            rescue_result = self._fuse_rescue(rescue_match, posture_status_dict, pose)
            class_name = rescue_result["class_name"]
            confidence = rescue_result["confidence"]
            if rescue_match is None and class_name == "normal_person":
                confidence = float(conf)
                
            pose.update(rescue_result)
            pose["rescue_match"] = {
                "is_victim": class_name == "victim_person",
                "is_high_conf": class_name == "victim_person" and confidence >= config.RESCUE_HIGH_CONF_THRESHOLD,
                "confidence": confidence,
                "class_name": class_name,
                "raw_class_name": (rescue_match or {}).get("class_name", "missing"),
                "raw_confidence": float((rescue_match or {}).get("confidence", 0.0)),
            }

            # Màng lọc bóng ma: Bỏ qua đồ vật vô tri vô giác, NHƯNG cứu nạn nhân thật hoặc người nằm sấp
            if max(box_w, box_h) > 60 and valid_kps < 3:
                # Nếu không ai công nhận nó là victim hay lying_person, thì vứt đi
                if class_name != "victim_person" and pose.get("posture_class") != "lying_person":
                    continue

            detections.append(RawDetection(object_id, person_box, class_name, confidence, pose))

        return detections

    # Tính độ tự tin tư thế dựa trên keypoints của Pose.
    # Trả về hệ số 0.0-1.0 dùng làm trọng số vote trong Late Fusion.
    @staticmethod
    def _pose_posture_confidence(pose: dict) -> float:
        status = str(pose.get("posture_status", "unknown") or "unknown")
        if status == "unknown":
            return 0.0
        confidence = str(pose.get("posture_confidence", "weak") or "weak")
        base = 0.80 if confidence == "strong" else 0.60
        kps_conf = pose.get("kps_conf")
        if kps_conf is not None:
            try:
                arr = np.asarray(kps_conf)
                visible_count = float(np.sum(arr >= 0.30))
                visible_ratio = min(1.0, visible_count / 9.0)  
                base *= 0.70 + 0.30 * visible_ratio
            except Exception:
                pass
        return float(clamp(base, 0.0, 1.0))

    # DUNG HỢP TƯ THẾ (Late Fusion): Kết hợp kết quả Posture model với
    # gợi ý từ Pose keypoints + tỷ lệ Bounding Box để vote ra tư thế cuối cùng
    # (Đứng/Ngồi/Nằm). Tư thế nào có tổng điểm cao nhất sẽ thắng.
    def _fuse_posture(self, posture_match: dict | None, pose: dict) -> dict:
        scores = {"standing": 0.0, "sitting": 0.0, "lying_or_fallen": 0.0}
        custom_status = "unknown"
        custom_conf = 0.0
        has_posture_match = posture_match is not None

        if has_posture_match:
            custom_status = {
                "standing_person": "standing",
                "sitting_person": "sitting",
                "lying_person": "lying_or_fallen",
                "stand": "standing",
                "sit": "sitting",
                "lie": "lying_or_fallen",
            }.get(normalize_posture_class(posture_match.get("class_name", "unknown")), "unknown")
            custom_conf = float(clamp(float(posture_match.get("confidence", 0.0)), 0.0, 1.0))
            if custom_status in scores:
                scores[custom_status] += custom_conf * config.POSTURE_MODEL_WEIGHT

            if custom_conf >= 0.55 and custom_status != "unknown":
                return {
                    "posture_status": custom_status,
                    "posture_class": {
                        "standing": "standing_person",
                        "sitting": "sitting_person",
                        "lying_or_fallen": "lying_person",
                    }.get(custom_status, "unknown"),
                    "posture_confidence": "strong",
                    "posture_fusion_score": round(custom_conf * config.POSTURE_MODEL_WEIGHT, 3),
                    "posture_model_status": custom_status,
                    "posture_model_confidence": round(custom_conf, 3),
                    "pose_posture_status": str(pose.get("posture_status", "unknown") or "unknown"),
                    "pose_posture_confidence": 0.0,
                    "posture_votes": scores,
                    "posture_match_found": True,
                }
        else:
            hint_status = {
                "standing_person": "standing",
                "sitting_person": "sitting",
                "lying_person": "lying_or_fallen",
                "stand": "standing",
                "sit": "sitting",
                "lie": "lying_or_fallen",
            }.get(str(pose.get("posture_hint", "unknown")), "unknown")
            hint_conf = float(pose.get("posture_hint_conf", 0.0))
            if hint_status in scores and hint_conf > 0:
                scores[hint_status] += hint_conf * config.POSTURE_MODEL_WEIGHT * 0.75

        pose_status = str(pose.get("posture_status", "unknown") or "unknown")
        pose_conf = self._pose_posture_confidence(pose)
        effective_pose_weight = config.POSE_POSTURE_WEIGHT if has_posture_match else min(0.50, config.POSE_POSTURE_WEIGHT + config.POSTURE_MODEL_WEIGHT * 0.35)
        if pose_status in scores:
            scores[pose_status] += pose_conf * effective_pose_weight

        pose_box = pose.get("bbox")
        if pose_box is not None and not has_posture_match:
            try:
                bx1, by1, bx2, by2 = pose_box
                bh = max(1, by2 - by1)
                bw = max(1, bx2 - bx1)
                ratio = bh / bw
                if 0.80 <= ratio < 1.65:
                    if scores["sitting"] > 0.15:
                        scores["sitting"] += 0.10
                elif ratio >= 1.65:
                    if scores["standing"] > 0.15:
                        scores["standing"] += 0.08
            except Exception:
                pass

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_status, best_score = ordered[0]
        second_score = ordered[1][1]
        margin = best_score - second_score
        min_conf = config.POSTURE_FUSION_MIN_CONFIDENCE if has_posture_match else (config.POSTURE_FUSION_MIN_CONFIDENCE - 0.08)
        min_margin = config.POSTURE_FUSION_MIN_MARGIN if has_posture_match else (config.POSTURE_FUSION_MIN_MARGIN * 0.5)
        if best_score < min_conf or margin < min_margin:
            best_status = "unknown"

        return {
            "posture_status": best_status,
            "posture_class": {
                "standing": "standing_person",
                "sitting": "sitting_person",
                "lying_or_fallen": "lying_person",
            }.get(best_status, "unknown"),
            "posture_confidence": "strong" if best_score >= 0.55 else "weak",
            "posture_fusion_score": round(best_score, 3),
            "posture_model_status": custom_status,
            "posture_model_confidence": round(custom_conf, 3),
            "pose_posture_status": pose_status,
            "pose_posture_confidence": round(pose_conf, 3),
            "posture_votes": {key: round(value, 3) for key, value in scores.items()},
            "posture_match_found": has_posture_match,
        }

    # Tính bằng chứng 'nghi nạn nhân' từ keypoints Pose.
    # Ví dụ: nằm ngã, vẫy tay, gục ngã → trả về hệ số 0.0-1.0 dùng boost điểm Rescue.
    @staticmethod
    def _pose_victim_evidence(pose: dict) -> float:
        evidence = 0.0
        if pose.get("is_lying"):
            evidence = max(evidence, 0.72)
        if pose.get("is_collapsed"):
            evidence = max(evidence, 0.82)
        if pose.get("is_immobile") and (pose.get("is_lying") or pose.get("is_collapsed")):
            evidence = max(evidence, 0.90)
        gesture = str(pose.get("gesture_candidate", "none") or "none")
        if gesture == "two_hands_waving":
            evidence = max(evidence, 0.90)
        elif gesture == "two_hands_raised":
            evidence = max(evidence, 0.65)
        elif gesture == "one_hand_raised":
            evidence = max(evidence, 0.25)
        return evidence

    # DUNG HỢP TRẠNG THÁI (Late Fusion): Kết hợp kết quả Rescue model với
    # bằng chứng từ Posture và Pose. Nếu Rescue nói Bình thường nhưng Pose/Posture
    # phát hiện dấu hiệu nguy kịch, cơ chế này sẽ lật ngược thành Victim.
    def _fuse_rescue(self, rescue_match: dict | None, posture: dict, pose: dict) -> dict:
        rescue_class = str((rescue_match or {}).get("class_name", "normal_person"))
        rescue_conf = float(clamp(float((rescue_match or {}).get("confidence", 0.0)), 0.0, 1.0))
        rescue_victim = rescue_conf if normalize_rescue_class(rescue_class) == "victim_person" else 0.0

        posture_victim = 0.0
        if posture.get("posture_status") == "lying_or_fallen":
            posture_victim = float(clamp(float(posture.get("posture_fusion_score", 0.0)) / config.POSTURE_MODEL_WEIGHT, 0.0, 1.0))
        pose_victim = self._pose_victim_evidence(pose)

        victim_probability = 1.0 - (
            (1.0 - rescue_victim)
            * (1.0 - 0.78 * posture_victim)
            * (1.0 - 0.65 * pose_victim)
        )
        class_name = "victim_person" if victim_probability >= config.VICTIM_FUSION_THRESHOLD else "normal_person"
        if class_name == "victim_person":
            confidence = victim_probability
        else:
            pose_conf = float(pose.get("confidence", 0.0))
            if normalize_rescue_class(rescue_class) == "normal_person" and rescue_conf > 0:
                confidence = max(rescue_conf, pose_conf)
            else:
                confidence = pose_conf

        return {
            "class_name": class_name,
            "confidence": float(clamp(confidence, 0.0, 1.0)),
            "victim_probability": round(victim_probability, 3),
            "rescue_victim_evidence": round(rescue_victim, 3),
            "posture_victim_evidence": round(posture_victim, 3),
            "pose_victim_evidence": round(pose_victim, 3),
            "source": "rescue+posture+pose",
        }

    # PHÂN TÍCH TOÀN DIỆN trạng thái 1 người từ 17 điểm keypoints.
    # Tính toán: Giơ tay (so Y cổ tay với vai), Vẫy tay (History Buffer),
    # Nằm/Ngã (tỷ lệ BBox + trục Vai-Hông), Gục ngã (3 yếu tố: đầu nghiêng,
    # thân nghiêng, thân ép), Đứng/Ngồi (Dot Product góc gập đầu gối),
    # Bất động (Center History displacement).
    def _analyze_person_state(self, box, kps_xy, kps_conf, box_id: int, current_time: float) -> dict:
        x1, y1, x2, y2 = box
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        reasons = []

        is_lying = False
        if box_w / box_h > 1.8:
            ls_early = get_keypoint(kps_xy, kps_conf, 5, min_conf=0.25)
            rs_early = get_keypoint(kps_xy, kps_conf, 6, min_conf=0.25)
            lw_early = get_keypoint(kps_xy, kps_conf, 9, min_conf=0.25)
            rw_early = get_keypoint(kps_xy, kps_conf, 10, min_conf=0.25)
            shoulder_y_avg = None
            if ls_early and rs_early:
                shoulder_y_avg = (ls_early[1] + rs_early[1]) / 2.0
            elif ls_early:
                shoulder_y_avg = ls_early[1]
            elif rs_early:
                shoulder_y_avg = rs_early[1]
            is_raising_now = False
            if shoulder_y_avg is not None:
                if (lw_early and lw_early[1] < shoulder_y_avg - 15) or (rw_early and rw_early[1] < shoulder_y_avg - 15):
                    is_raising_now = True
            nose_check = get_keypoint(kps_xy, kps_conf, 0, min_conf=0.3)
            if not is_raising_now:
                if nose_check is None or nose_check[1] >= y1 + box_h * 0.45:
                    is_lying = True
                    reasons.append("nam/nga")

        left_shoulder = get_keypoint(kps_xy, kps_conf, 5)
        right_shoulder = get_keypoint(kps_xy, kps_conf, 6)
        left_wrist = get_keypoint(kps_xy, kps_conf, 9)
        right_wrist = get_keypoint(kps_xy, kps_conf, 10)
        nose = get_keypoint(kps_xy, kps_conf, 0)
        left_hip = get_keypoint(kps_xy, kps_conf, 11)
        right_hip = get_keypoint(kps_xy, kps_conf, 12)
        left_knee = get_keypoint(kps_xy, kps_conf, 13)
        right_knee = get_keypoint(kps_xy, kps_conf, 14)
        left_ankle = get_keypoint(kps_xy, kps_conf, 15)
        right_ankle = get_keypoint(kps_xy, kps_conf, 16)

        is_raising_hand = False
        is_one_hand_raised = False
        is_two_hands_raised = False
        is_waving = False
        gesture_candidate = "none"
        left_wrist_sample = None
        right_wrist_sample = None
        shoulder_points = [p for p in (left_shoulder, right_shoulder) if p is not None]
        if shoulder_points:
            shoulder_y = sum(p[1] for p in shoulder_points) / len(shoulder_points)
            raised_count = 0
            left_above = False
            right_above = False
            if left_wrist is not None and left_wrist[1] < shoulder_y - config.HAND_RAISE_MARGIN_PIXELS:
                raised_count += 1
                left_above = True
            if right_wrist is not None and right_wrist[1] < shoulder_y - config.HAND_RAISE_MARGIN_PIXELS:
                raised_count += 1
                right_above = True

            anchor_x, anchor_y = cx, cy
            if nose is not None:
                anchor_x, anchor_y = nose
            elif shoulder_points:
                anchor_x = sum(p[0] for p in shoulder_points) / len(shoulder_points)
                anchor_y = sum(p[1] for p in shoulder_points) / len(shoulder_points)

            if left_wrist is not None:
                rel_x = (left_wrist[0] - anchor_x) / box_w
                rel_y = (left_wrist[1] - anchor_y) / box_h
                left_wrist_sample = (current_time, float(rel_x), float(rel_y), bool(left_above))
                self.left_wrist_history[box_id].append(left_wrist_sample)
            if right_wrist is not None:
                rel_x = (right_wrist[0] - anchor_x) / box_w
                rel_y = (right_wrist[1] - anchor_y) / box_h
                right_wrist_sample = (current_time, float(rel_x), float(rel_y), bool(right_above))
                self.right_wrist_history[box_id].append(right_wrist_sample)

            left_wave = detect_wrist_waving(self.left_wrist_history[box_id], current_time)
            right_wave = detect_wrist_waving(self.right_wrist_history[box_id], current_time)
            if raised_count >= 2:
                is_raising_hand = True
                is_two_hands_raised = True
                gesture_candidate = "two_hands_raised"
                reasons.append("2 tay gio len")
            elif raised_count == 1:
                is_raising_hand = True
                is_one_hand_raised = True
                gesture_candidate = "one_hand_raised"
                reasons.append("gio tay")

            if raised_count >= 2 and left_wave and right_wave:
                is_waving = True
                gesture_candidate = "two_hands_waving"
                reasons.append("2 tay vay cau cuu")

        hip_points = [p for p in (left_hip, right_hip) if p is not None]
        knee_points = [p for p in (left_knee, right_knee) if p is not None]
        ankle_points = [p for p in (left_ankle, right_ankle) if p is not None]
        is_collapsed = False
        is_sitting_pose = False
        is_standing_pose = False
        posture_confidence = "weak"
        if hip_points and shoulder_points:
            avg_hip_y = sum(p[1] for p in hip_points) / len(hip_points)
            avg_hip_x = sum(p[0] for p in hip_points) / len(hip_points)
            avg_shoulder_y = sum(p[1] for p in shoulder_points) / len(shoulder_points)
            avg_shoulder_x = sum(p[0] for p in shoulder_points) / len(shoulder_points)
            torso_height = abs(avg_hip_y - avg_shoulder_y)
            spine_x_span = abs(avg_hip_x - avg_shoulder_x)

            if len(shoulder_points) == 2:
                shoulder_width = abs(shoulder_points[0][0] - shoulder_points[1][0])
            else:
                shoulder_width = max(10.0, box_w * 0.3)

            body_points_torso = shoulder_points + hip_points + knee_points + ankle_points
            shoulder_hip_gap = abs(avg_hip_y - avg_shoulder_y)
            torso_is_horizontal = shoulder_hip_gap < box_h * 0.20
            if len(body_points_torso) >= 4:
                body_x_span = max(p[0] for p in body_points_torso) - min(p[0] for p in body_points_torso)
                body_y_span = max(p[1] for p in body_points_torso) - min(p[1] for p in body_points_torso)
                if body_x_span > max(box_w * 0.35, body_y_span * config.LYING_KEYPOINT_RATIO_THRESHOLD) and torso_is_horizontal:
                    is_lying = True
                    reasons.append("truc co the nam ngang")
            if spine_x_span > max(box_w * 0.25, torso_height * 1.35) and torso_height < box_h * 0.35 and torso_is_horizontal:
                is_lying = True
                reasons.append("than nguoi nam ngang")

            if knee_points and not is_lying:
                avg_knee_y = sum(p[1] for p in knee_points) / len(knee_points)
                avg_knee_x = sum(p[0] for p in knee_points) / len(knee_points)
                head_to_hip_y = (avg_hip_y - nose[1]) if nose else (torso_height * 1.3)
                head_to_hip_y = max(1.0, head_to_hip_y)
                upper_leg_y = max(1.0, avg_knee_y - avg_hip_y)
                vertical_ratio = box_h / max(1.0, float(box_w))

                if ankle_points:
                    avg_ankle_y = sum(p[1] for p in ankle_points) / len(ankle_points)
                    avg_ankle_x = sum(p[0] for p in ankle_points) / len(ankle_points)
                    
                    vec_hip_knee = (avg_knee_x - avg_hip_x, avg_knee_y - avg_hip_y)
                    vec_knee_ankle = (avg_ankle_x - avg_knee_x, avg_ankle_y - avg_knee_y)
                    
                    dot_product = vec_hip_knee[0]*vec_knee_ankle[0] + vec_hip_knee[1]*vec_knee_ankle[1]
                    mag_hk = math.hypot(vec_hip_knee[0], vec_hip_knee[1])
                    mag_ka = math.hypot(vec_knee_ankle[0], vec_knee_ankle[1])
                    
                    knee_angle = 180.0
                    if mag_hk > 0 and mag_ka > 0:
                        cos_val = max(-1.0, min(1.0, dot_product / (mag_hk * mag_ka)))
                        knee_angle = math.degrees(math.acos(cos_val))
                    
                    if knee_angle > 45.0:
                        is_sitting_pose = True
                        posture_confidence = "strong"
                        reasons.append(f"dang ngoi: goc dau goi gap {int(knee_angle)} do")
                    elif upper_leg_y <= head_to_hip_y * 0.35 and vertical_ratio < 2.6:
                        is_sitting_pose = True
                        posture_confidence = "weak"
                        reasons.append("co the ngoi: dui ngan du thay co chan")
                    else:
                        is_standing_pose = True
                        posture_confidence = "strong"
                        reasons.append(f"dang dung: chan duoi thang (goc lech {int(knee_angle)} do)")
                else:
                    if upper_leg_y <= head_to_hip_y * 0.35 and vertical_ratio < 2.6:
                        is_sitting_pose = True
                        posture_confidence = "strong"
                        reasons.append("dang ngoi: dui ngan gap goc (khuat co chan)")
                    elif upper_leg_y >= head_to_hip_y * 0.35 and vertical_ratio >= 1.1 and not is_sitting_pose:
                        is_standing_pose = True
                        posture_confidence = "strong"
                        reasons.append("dang dung: dui dai doc xuong (khuat co chan)")

            head_tilted = False
            if nose is not None:
                if abs(nose[0] - avg_shoulder_x) > max(box_w * 0.15, shoulder_width * 0.65):
                    head_tilted = True
                elif nose[1] - avg_shoulder_y > max(box_h * 0.08, torso_height * 0.2):
                    head_tilted = True
            body_leaning = False
            if spine_x_span > max(box_w * 0.20, torso_height * 0.45):
                body_leaning = True
                    
            torso_crushed = torso_height / max(1.0, float(box_h)) < 0.15
            if head_tilted or torso_crushed or body_leaning:
                is_collapsed = True
                reasons.append("guc dau / nghieng / sup do")

        history = self.movement_history[box_id]
        history.append((current_time, cx, cy))
        while history and current_time - history[0][0] > config.STILLNESS_SECONDS:
            history.popleft()

        is_immobile = False
        if len(history) >= 5:
            times = [p[0] for p in history]
            xs = [p[1] for p in history]
            ys = [p[2] for p in history]
            duration = max(times) - min(times)
            if duration >= config.STILLNESS_SECONDS - 0.5 and max(xs) - min(xs) < config.CENTER_JITTER_THRESHOLD and max(ys) - min(ys) < config.CENTER_JITTER_THRESHOLD:
                is_immobile = True
                if is_lying or is_collapsed:
                    reasons.append("bat dong")

        if is_lying and is_immobile:
            reasons.append("nam lau")
        if is_raising_hand and is_immobile:
            reasons.append("gio tay + bat dong")

        posture_status = "unknown"
        if is_standing_pose:
            posture_status = "standing"
        elif is_sitting_pose:
            posture_status = "sitting"
        elif is_lying:
            posture_status = "lying_or_fallen"

        state = "BINH_THUONG"
        if is_waving or is_two_hands_raised or (is_lying and is_collapsed):
            state = "CAN_CUU_GIUP"
        elif is_lying or is_collapsed or is_one_hand_raised or is_immobile:
            state = "NGHI_NGO"

        return {
            "source": "pose_rescue",
            "state": state,
            "danger_score": 0,
            "reasons": reasons,
            "is_lying": is_lying,
            "is_immobile": is_immobile,
            "is_raising_hand": is_raising_hand,
            "is_one_hand_raised": is_one_hand_raised,
            "is_two_hands_raised": is_two_hands_raised,
            "is_waving": is_waving,
            "is_collapsed": is_collapsed,
            "posture_status": posture_status,
            "posture_confidence": posture_confidence,
            "posture_hint": "standing" if is_standing_pose else ("sitting" if is_sitting_pose else "unknown"),
            "posture_hint_conf": 0.8 if posture_confidence == "strong" else 0.4,
            "gesture_candidate": gesture_candidate,
            "left_wrist_sample": left_wrist_sample,
            "right_wrist_sample": right_wrist_sample,
        }

    # Tạo dữ liệu giả lập mục tiêu khi không có model YOLO để test giao diện.
    def _simulate(self, frame, timestamp: float) -> list[RawDetection]:
        h, w = frame.shape[:2]
        cx = int(w * (0.45 + 0.12 * math.sin(timestamp / 3.0)))
        cy = int(h * 0.54)
        return [
            RawDetection(
                1,
                (cx - 95, cy - 42, cx + 95, cy + 42),
                "victim_person",
                0.82,
                {
                    "source": "simulated",
                    "state": "CAN_CUU_GIUP",
                    "danger_score": 4,
                    "reasons": ["demo victim"],
                    "is_lying": True,
                    "is_immobile": False,
                    "is_raising_hand": False,
                    "is_one_hand_raised": False,
                    "is_two_hands_raised": False,
                    "is_waving": False,
                    "is_collapsed": False,
                    "posture_status": "lying_or_fallen",
                    "gesture_candidate": "none",
                    "left_wrist_sample": None,
                    "right_wrist_sample": None,
                    "rescue_match": {"is_high_conf": True},
                },
            )
        ]


