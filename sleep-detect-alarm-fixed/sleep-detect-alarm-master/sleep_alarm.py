import cv2
import mediapipe as mp
import numpy as np
import time
import os
import threading

try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[WARNING] pygame not found. Install it with:  pip install pygame")

# ── Tunable settings ────────────────────────────────────────────────────────
EAR_THRESHOLD      = 0.22   # below this → eyes are considered closed
EYE_CLOSED_SECONDS = 2.5    # seconds before alarm fires

# Alarm sound file — relative to this script's folder
# Rename your mp3 to "alarm.mp3" and keep it next to this file,
# OR set an absolute path here, e.g. r"C:\Users\you\alarm.mp3"
ALARM_SOUND_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "dragon-studio-censor-beep-3-372460.mp3")
# ────────────────────────────────────────────────────────────────────────────

LEFT_EYE_INDICES  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33,  160, 158, 133, 153, 144]


def eye_aspect_ratio(landmarks, eye_indices, frame_w, frame_h):
    pts = [np.array([landmarks[i].x * frame_w, landmarks[i].y * frame_h])
           for i in eye_indices]
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C)


def play_alarm():
    if not PYGAME_AVAILABLE:
        print("\a[ALARM] Eyes closed too long! (No sound – pygame not installed)")
        return
    if not os.path.isfile(ALARM_SOUND_FILE):
        print(f"[ALARM] Sound file not found: {ALARM_SOUND_FILE}")
        print("[ALARM] Place your mp3 next to this script and update ALARM_SOUND_FILE.")
        print("\a")
        return
    try:
        pygame.mixer.music.load(ALARM_SOUND_FILE)
        pygame.mixer.music.play(-1)
    except Exception as e:
        print(f"[ALARM] Could not play sound: {e}")


def stop_alarm():
    if not PYGAME_AVAILABLE:
        return
    try:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
    except Exception:
        pass


def draw_status_overlay(frame, alarm_active, elapsed, ear_avg):
    h, w = frame.shape[:2]

    if alarm_active:
        color    = (0, 0, 220)
        bg_color = (0, 0, 180)
        label    = "SLEEPING! WAKE UP!"
    else:
        color    = (0, 200, 50)
        bg_color = (0, 150, 40)
        label    = "AWAKE"

    # Pulsing red overlay when alarm is active
    if alarm_active:
        pulse   = int(abs(np.sin(time.time() * 4)) * 80)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
        alpha = 0.20 + pulse / 1000
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    # Status pill
    pill_w, pill_h = 340, 54
    pill_x = (w - pill_w) // 2
    pill_y = 14
    cv2.rectangle(frame, (pill_x, pill_y),
                  (pill_x + pill_w, pill_y + pill_h), bg_color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (pill_x, pill_y),
                  (pill_x + pill_w, pill_y + pill_h), color, 2, cv2.LINE_AA)
    cv2.putText(frame, label, (pill_x + 20, pill_y + 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # Status dot
    dot_x = pill_x - 30
    dot_y = pill_y + pill_h // 2
    cv2.circle(frame, (dot_x, dot_y), 14, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (dot_x, dot_y), 14, (255, 255, 255), 2, cv2.LINE_AA)

    # Bottom info bar
    bar_h = 50
    bar_y = h - bar_h
    cv2.rectangle(frame, (0, bar_y), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, f"EAR: {ear_avg:.3f}  (thresh: {EAR_THRESHOLD})",
                (14, bar_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Eyes closed: {elapsed:.1f}s / {EYE_CLOSED_SECONDS}s",
                (14, bar_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # Progress bar
    if elapsed > 0 and not alarm_active:
        progress    = min(elapsed / EYE_CLOSED_SECONDS, 1.0)
        bar_fill_w  = int((w - 28) * progress)
        cv2.rectangle(frame, (14, bar_y + 45),
                      (14 + bar_fill_w, bar_y + 49), (0, 165, 255), -1, cv2.LINE_AA)

    return frame


def draw_face_box(frame, landmarks, color, w, h):
    xs  = [int(lm.x * w) for lm in landmarks]
    ys  = [int(lm.y * h) for lm in landmarks]
    pad = 10
    x1  = max(0, min(xs) - pad)
    y1  = max(0, min(ys) - pad)
    x2  = min(w, max(xs) + pad)
    y2  = min(h, max(ys) + pad)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    corner = 18
    thick  = 3
    for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (cx, cy), (cx + dx*corner, cy), color, thick, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy*corner), color, thick, cv2.LINE_AA)


def open_camera():
    """Try camera index 0, then 1 — returns the first one that opens."""
    for index in (0, 1):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"[INFO] Camera opened at index {index}")
            return cap
        cap.release()
    return None


def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh    = mp_face_mesh.FaceMesh(
        max_num_faces        = 1,
        refine_landmarks     = True,
        min_detection_confidence = 0.6,
        min_tracking_confidence  = 0.6,
    )

    # FIX: try index 0 first, fallback to 1
    cap = open_camera()
    if cap is None:
        print("[ERROR] Cannot open any webcam (tried index 0 and 1).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    eye_closed_start = None
    alarm_active     = False
    alarm_thread     = None

    print("=" * 60)
    print("  Sleep Detector — press Q to quit")
    print(f"  Alarm fires after {EYE_CLOSED_SECONDS}s of closed eyes")
    if PYGAME_AVAILABLE:
        print(f"  Alarm sound: {ALARM_SOUND_FILE}")
        if not os.path.isfile(ALARM_SOUND_FILE):
            print("  [WARNING] Sound file not found — will beep only")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Frame read failed, retrying...")
            continue

        frame  = cv2.flip(frame, 1)
        h, w   = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        ear_avg      = 1.0
        elapsed      = 0.0
        face_detected = False

        if results.multi_face_landmarks:
            face_detected = True
            lms   = results.multi_face_landmarks[0].landmark
            ear_l = eye_aspect_ratio(lms, LEFT_EYE_INDICES,  w, h)
            ear_r = eye_aspect_ratio(lms, RIGHT_EYE_INDICES, w, h)
            ear_avg = (ear_l + ear_r) / 2.0

            eyes_closed = ear_avg < EAR_THRESHOLD

            if eyes_closed:
                if eye_closed_start is None:
                    eye_closed_start = time.time()
                elapsed = time.time() - eye_closed_start
                if elapsed >= EYE_CLOSED_SECONDS and not alarm_active:
                    alarm_active  = True
                    alarm_thread  = threading.Thread(target=play_alarm, daemon=True)
                    alarm_thread.start()
            else:
                eye_closed_start = None
                if alarm_active:
                    alarm_active = False
                    stop_alarm()

            face_color = (0, 0, 220) if alarm_active else (0, 220, 60)
            draw_face_box(frame, lms, face_color, w, h)

        else:
            eye_closed_start = None
            if alarm_active:
                alarm_active = False
                stop_alarm()

        frame = draw_status_overlay(frame, alarm_active, elapsed, ear_avg)

        if not face_detected:
            cv2.putText(frame, "No face detected",
                        (w // 2 - 130, h // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 180, 255), 2, cv2.LINE_AA)

        cv2.imshow("Sleep Alarm Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_alarm()
    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
