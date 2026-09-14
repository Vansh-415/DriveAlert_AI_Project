import time
from collections import deque
import cv2, numpy as np


class DrowsinessDetector:
    """Face & bilateral eye detection, CNN inference, and temporal smoothing."""

    def __init__(self, model):
        self.model = model
        shape = model.input_shape
        self.target_h = shape[1] if shape[1] is not None else 96
        self.target_w = shape[2] if shape[2] is not None else 96
        self.channels = shape[3] if len(shape) > 3 and shape[3] is not None else 3

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        self.history = deque(maxlen=20)
        self.smoothed_state, self.consecutive_drowsy, self.fps, self.prev_time = "Alert", 0, 0.0, time.time()

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        now = time.time()
        if (dt := now - self.prev_time) > 0:
            self.fps = 0.85 * self.fps + 0.15 * (1.0 / dt)
        self.prev_time = now

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.15, 4, minSize=(60, 60))

        if len(faces) == 0:
            self.consecutive_drowsy = 0
            cv2.putText(frame_bgr, "No face detected", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
            cv2.putText(frame_bgr, f"Drowsy: 0 | FPS: {self.fps:.1f}", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            return frame_bgr

        fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        roi_gray, roi_color = gray[fy:fy + fh, fx:fx + fw], frame_bgr[fy:fy + fh, fx:fx + fw]

        # Scan eyes in the upper face band (20% - 55% height) to avoid mouth/nose false positives
        y1, y2 = int(fh * 0.20), int(fh * 0.55)
        mid_x, ey_w, ey_h = int(fw * 0.5), int(fw * 0.30), int(fh * 0.28)

        # Detect left and right eyes separately for high accuracy
        l_eyes = self.eye_cascade.detectMultiScale(roi_gray[y1:y2, :mid_x], 1.1, 3, minSize=(16, 16))
        r_eyes = self.eye_cascade.detectMultiScale(roi_gray[y1:y2, mid_x:], 1.1, 3, minSize=(16, 16))

        l_box = (max(l_eyes, key=lambda b: b[2]*b[3])[0], max(l_eyes, key=lambda b: b[2]*b[3])[1] + y1, max(l_eyes, key=lambda b: b[2]*b[3])[2], max(l_eyes, key=lambda b: b[2]*b[3])[3]) if len(l_eyes) else (int(fw * 0.14), y1, ey_w, ey_h)
        r_box = (max(r_eyes, key=lambda b: b[2]*b[3])[0] + mid_x, max(r_eyes, key=lambda b: b[2]*b[3])[1] + y1, max(r_eyes, key=lambda b: b[2]*b[3])[2], max(r_eyes, key=lambda b: b[2]*b[3])[3]) if len(r_eyes) else (int(fw * 0.56), y1, ey_w, ey_h)

        eye_statuses = []
        for (ex, ey, ew, eh) in [l_box, r_box]:
            pw, ph = int(ew * 0.12), int(eh * 0.12)
            crop = roi_color[max(0, ey - ph):min(fh, ey + eh + ph), max(0, ex - pw):min(fw, ex + ew + pw)]
            if crop.size == 0:
                continue
            img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if self.channels == 3 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            inp = np.expand_dims(cv2.resize(img, (self.target_w, self.target_h)).astype(np.float32) / 255.0, axis=0)
            pred = float(self.model.predict(inp, verbose=0)[0][0])
            status = "Alert" if pred > 0.50 else "Warning" if pred >= 0.40 else "Drowsy"
            eye_statuses.append(status)
            color = (0, 255, 0) if status == "Alert" else (0, 200, 255) if status == "Warning" else (0, 0, 255)
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), color, 2)
            cv2.putText(roi_color, "Open" if status == "Alert" else "Closed", (ex, max(12, ey - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        # ONE-EYE OPEN LOGIC: If at least one eye is open (Alert), the person is awake
        if "Alert" in eye_statuses:
            frame_label = "Alert"
        elif any(s == "Warning" for s in eye_statuses):
            frame_label = "Warning"
        elif all(s == "Drowsy" for s in eye_statuses) and len(eye_statuses) > 0:
            frame_label = "Drowsy"
        else:
            frame_label = "Alert"

        # TEMPORAL SMOOTHING LOGIC:
        # Buffer blinks over rolling 20 frames. Only alert if >= 15 of last 20 frames show drowsiness.
        self.history.append(frame_label)
        dw_count = sum(1 for label in self.history if label in ("Drowsy", "Warning"))
        if dw_count >= 15:
            self.smoothed_state, self.consecutive_drowsy = "Drowsy", self.consecutive_drowsy + 1
        elif dw_count >= 6:
            self.smoothed_state, self.consecutive_drowsy = "Warning", 0
        else:
            self.smoothed_state, self.consecutive_drowsy = "Alert", 0

        # Visual alert: Green for Alert; Red box + top banner for Drowsy
        if self.smoothed_state == "Drowsy":
            cv2.rectangle(frame_bgr, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 3)
            cv2.rectangle(frame_bgr, (0, 0), (w, 52), (0, 0, 255), -1)
            tx = max(10, (w - cv2.getTextSize("DROWSY - WAKE UP!", cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0][0]) // 2)
            cv2.putText(frame_bgr, "DROWSY - WAKE UP!", (tx, 36), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        else:
            box_col = (0, 200, 255) if self.smoothed_state == "Warning" else (0, 255, 0)
            cv2.rectangle(frame_bgr, (fx, fy), (fx + fw, fy + fh), box_col, 2)

        cv2.putText(frame_bgr, f"Drowsy Frames: {self.consecutive_drowsy} | FPS: {self.fps:.1f}", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame_bgr
