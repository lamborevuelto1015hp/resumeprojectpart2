"""Standalone compact ADAS demo. Run: python ad_compact.py.

Uses the same camera/ML/GUI dependencies as ad.py (MediaPipe 0.10.14).
Optional CARLA server: CARLA_HOST=127.0.0.1, CARLA_PORT=2000.
YOLO annotates the road view; simulator actors supply lead distances.
"""

import torch
from ultralytics import YOLO

import csv
import ctypes
import math
import os
import random
import re
from statistics import median
import threading
import time
import traceback
from datetime import datetime

import cv2
import mediapipe as mp          # pip install mediapipe==0.10.14
import numpy as np
import speech_recognition as sr
import wx

try:
    import carla
except ImportError:
    carla = None

LOG_FILE = "adas_log_v3.csv"

# --- Driver monitoring ---
EAR_THRESHOLD = 0.21            # upper bound for calibrated eye-closure threshold
EAR_CLOSED_RATIO = 0.60         # fraction of neutral open-eye EAR
DROWSY_SECONDS = 2.0            # eyes closed this long -> drowsy
DISTRACT_SECONDS = 2.0          # eyes off road this long -> distracted
OFF_ROAD_YAW_DEG = 25.0         # head+gaze turned further than this = off road
OFF_ROAD_PITCH_DOWN_DEG = 20.0  # head tilted down further than this = off road (phone)
CALIBRATION_FRAMES = 30         # ~1 s of "look straight at the screen"
GAZE_DEG_PER_UNIT = 100.0       # iris offset (fraction of eye width) -> degrees

# Fixed chase view, in vehicle coordinates (x forward, y right, z up).
CHASE_X, CHASE_Y, CHASE_Z = -6.0, 0.0, 3.2
CHASE_PITCH = -15.0
HEAD_STEER_DEADZONE = 8.0       # ignore small head movements
HEAD_STEER_FULL = 40.0
HEAD_MAX_STEER = 0.45
HEAD_STEER_RATE = 0.6           # maximum steering change per second
HEAD_THROTTLE_RATE = 0.35
STRAIGHT_ENTER = (10.0, 12.0, 12.0)  # yaw, pitch, horizontal gaze (degrees)
STRAIGHT_EXIT = (15.0, 18.0, 18.0)   # wider exit limits prevent boundary flicker
STRAIGHT_HOLD_S = 0.75               # look forward continuously before resuming
HEAD_PEDAL_DEADZONE = 12.0       # pitch degrees before throttle/brake responds
HEAD_PEDAL_FULL = 35.0          # pitch degrees for full pedal input

# --- Simulation ---
NUM_NPC = 15
BRAKE_HOLD_S = 1.5              # keep braking at least this long before resuming autopilot
LANE_HALF_WIDTH = 2.0           # m, for "vehicle ahead" check
CAR_LENGTH_M = 4.5              # subtracted from centre-to-centre distance
ENABLE_YOLO = True

# Head pose model (generic 3D face points, mm) and matching MediaPipe landmarks.
# Image-left/right refers to the RAW (un-mirrored) camera frame.
POSE_IDX = [1, 152, 33, 263, 61, 291]   # nose, chin, eye L, eye R, mouth L, mouth R
MODEL_3D = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)], dtype=np.float64)
YAW_SIGN, PITCH_SIGN = 1.0, -1.0        # verified: yaw+ = user turns right, pitch+ = looks up

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def beep_async(freq=1000, ms=300):
    def _b():
        try:
            import winsound
            winsound.Beep(freq, ms)
        except ImportError:
            print("\a", end="", flush=True)
    threading.Thread(target=_b, daemon=True).start()

def eye_aspect_ratio(lm, idx, w, h):
    p = [(lm[i].x * w, lm[i].y * h) for i in idx]
    return (math.dist(p[1], p[5]) + math.dist(p[2], p[4])) / (2.0 * math.dist(p[0], p[3]) + 1e-6)

def head_pose(lm, w, h):
    """Returns (yaw, pitch) in degrees. yaw+ = user turns to THEIR right, pitch+ = looks up."""
    pts = np.array([(lm[i].x * w, lm[i].y * h) for i in POSE_IDX], dtype=np.float64)
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_3D, pts, cam, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    pitch, yaw, _roll = cv2.RQDecomp3x3(cv2.Rodrigues(rvec)[0])[0]
    if pitch > 90:      # frontal face decodes near +/-180 because of the model/camera axis flip
        pitch -= 180
    elif pitch < -90:
        pitch += 180
    return YAW_SIGN * yaw, PITCH_SIGN * pitch

def _iris_ratio(lm, iris, a, b):
    lo, hi = sorted((lm[a].x, lm[b].x))
    return (lm[iris].x - lo) / (hi - lo + 1e-6)

def gaze_offset(lm):
    """>0 = eyes look toward the user's right (iris moves to image-left in the raw frame)."""
    r = (_iris_ratio(lm, 468, 33, 133) + _iris_ratio(lm, 473, 362, 263)) / 2.0
    return 0.5 - r

class FrameWorker(threading.Thread):
    """A stoppable worker publishing its latest frame under a lock."""
    def __init__(self):
        super().__init__(daemon=True)
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._frame = None

    def stop(self):
        self._stop_evt.set()
        if self.is_alive():
            self.join(timeout=2)

    def get_frame(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

class DriverMonitor:
    """Separate sustained eye closure from looking away or unreliable head angles."""
    def __init__(self):
        self.closed_since = self.away_since = None
        self.threshold = EAR_THRESHOLD

    def update(self, face, calibrated, yaw, pitch, gaze, ear, neutral_ear, now):
        self.threshold = min(EAR_THRESHOLD, neutral_ear * EAR_CLOSED_RATIO) if neutral_ear else EAR_THRESHOLD
        reliable_eyes = face and calibrated and abs(yaw) < 35 and abs(pitch) < 25
        closed = reliable_eyes and ear is not None and ear < self.threshold
        away = not face or (calibrated and (
            abs(yaw + gaze) > OFF_ROAD_YAW_DEG or abs(pitch) > OFF_ROAD_PITCH_DOWN_DEG))
        self.closed_since = (now if self.closed_since is None else self.closed_since) if closed else None
        self.away_since = (now if self.away_since is None else self.away_since) if away else None
        if self.closed_since is not None and now - self.closed_since >= DROWSY_SECONDS:
            return "drowsy"
        if self.away_since is not None and now - self.away_since >= DISTRACT_SECONDS:
            return "distracted"
        return "normal"


class StraightDriveGate:
    """Require a stable forward pose before allowing CARLA autopilot to move."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.active, self.since = False, None

    def allow(self, snap):
        values = [snap.get(name, math.nan) for name in ("yaw", "pitch", "gaze_yaw")]
        limits = STRAIGHT_EXIT if self.active else STRAIGHT_ENTER
        forward = (snap.get("face") and snap.get("calibrated") and snap.get("state") == "normal"
                   and all(math.isfinite(v) and abs(v) <= limit for v, limit in zip(values, limits)))
        if not forward:
            self.reset()
            return False
        now = time.monotonic()
        self.since = now if self.since is None else self.since
        self.active = self.active or now - self.since >= STRAIGHT_HOLD_S
        return self.active


class DriverTracker(FrameWorker):
    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.error = None
        self._snap = dict(state="unavailable", face=False, yaw=0.0, pitch=0.0,
                          gaze_yaw=0.0, ear=None, calibrated=False)
        self.monitor = DriverMonitor()
        self._cal = []
        self._neutral = None
        self._updated_at = 0.0
        self._recenter = threading.Event()

    def recalibrate(self):
        self._recenter.set()
        with self._lock:
            self._snap.update(face=False, calibrated=False)

    def snapshot(self):
        with self._lock:
            snap = dict(self._snap)
            if self.error or self._recenter.is_set() or time.monotonic() - self._updated_at > 0.5:
                snap.update(state="unavailable", face=False, yaw=0.0, pitch=0.0, gaze_yaw=0.0)
            return snap

    def run(self):
        try:
            self._run_tracker()
        except Exception as e:
            self.error = f"Driver tracking failed: {type(e).__name__}: {e}"
            print(self.error)
            traceback.print_exc()
        finally:
            cap = getattr(self, "_cap", None)
            if cap is not None:
                cap.release()

    def _run_tracker(self):
        cap = cv2.VideoCapture(self.camera_index)
        self._cap = cap
        if not cap.isOpened():
            self.error = f"Could not open camera {self.camera_index}"
            return

        with mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5) as mesh:
            while not self._stop_evt.is_set():
                ok, raw = cap.read()
                if not ok:
                    with self._lock:
                        self._snap.update(state="unavailable", face=False, yaw=0.0, pitch=0.0)
                    self.monitor = DriverMonitor()
                    self._stop_evt.wait(0.02)
                    continue
                if self._recenter.is_set():
                    self._recenter.clear()
                    self._cal, self._neutral = [], None
                    self.monitor = DriverMonitor()
                h, w = raw.shape[:2]
                res = mesh.process(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))   # un-mirrored frame
                now = time.monotonic()
                if now - self._updated_at > 0.5:
                    self.monitor = DriverMonitor()

                face, ear, yaw, pitch, gaze_yaw = False, None, 0.0, 0.0, 0.0
                lm = None
                if res.multi_face_landmarks:
                    lm = res.multi_face_landmarks[0].landmark
                    pose = head_pose(lm, w, h)
                    if pose is not None:
                        face = True
                        raw_yaw, raw_pitch = pose
                        gaze_x = gaze_offset(lm)
                        ear = (eye_aspect_ratio(lm, RIGHT_EYE, w, h) +
                               eye_aspect_ratio(lm, LEFT_EYE, w, h)) / 2.0

                        if self._neutral is None:           # auto-calibrate on first frames
                            self._cal.append((raw_yaw, raw_pitch, gaze_x, ear))
                            if len(self._cal) >= CALIBRATION_FRAMES:
                                self._neutral = tuple(median(values) for values in zip(*self._cal))
                        n = self._neutral
                        if n is not None:
                            yaw, pitch = raw_yaw - n[0], raw_pitch - n[1]
                            gaze_yaw = (gaze_x - n[2]) * GAZE_DEG_PER_UNIT

                calibrated = self._neutral is not None
                state = self.monitor.update(face, calibrated, yaw, pitch, gaze_yaw, ear,
                                            self._neutral[3] if calibrated else None, now)

                # mirrored preview with overlay
                disp = cv2.flip(raw, 1)
                if lm is not None:
                    for i in RIGHT_EYE + LEFT_EYE:
                        cv2.circle(disp, (w - int(lm[i].x * w), int(lm[i].y * h)), 1, (0, 255, 0), -1)
                color = (0, 255, 0) if state == "normal" else (0, 0, 255)
                cv2.putText(disp, f"Driver: {state.upper()}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                if not calibrated:
                    cv2.putText(disp, "Calibrating - look straight at the screen", (20, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                elif face:
                    cv2.putText(disp, f"Yaw {yaw:+.0f}  Pitch {pitch:+.0f}  Gaze {gaze_yaw:+.0f}  EAR {ear:.2f}",
                                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                else:
                    cv2.putText(disp, "No face detected", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                with self._lock:
                    self._snap = dict(state=state, face=face, yaw=yaw, pitch=pitch,
                                      gaze_yaw=gaze_yaw, ear=ear, ear_threshold=self.monitor.threshold, calibrated=calibrated)
                    self._frame = disp
                    self._updated_at = time.monotonic()

def evaluate_adas(speed_kmh, distance_m, obstacle, road, driver_state):
    """Return TTC, heuristic risk and decision; infinity means no detected lead."""
    if not math.isfinite(speed_kmh) or speed_kmh < 0 or math.isnan(distance_m) or distance_m <= 0:
        raise ValueError("Speed must be finite and nonnegative; distance must be positive")
    speed = speed_kmh / 3.6
    ttc = distance_m / speed if speed > 0 else math.inf
    risk = speed / ttc * {"Dry": 1.0, "Wet": 1.5, "Poor": 2.0}.get(road, 1.0)
    risk *= {"Animal": 1.3, "Pothole": 0.8}.get(obstacle, 1.0)
    risk += {"drowsy": 7, "distracted": 4}.get(driver_state, 0)
    decision = next((name for threshold, name in [
        (25, "EMERGENCY BRAKE"), (15, "GRADUAL BRAKE + ALERT"),
        (8, "PRE-BRAKE + WARNING")] if risk > threshold), "MONITOR")
    return round(ttc, 2), round(risk, 2), decision

def brake_level(decision, state, auto):
    """Auto mode: obey the ADAS decision. Manual what-if mode: only react to the driver."""
    level = 0.0
    if auto:
        level = {"EMERGENCY BRAKE": 1.0, "GRADUAL BRAKE + ALERT": 0.6,
                 "PRE-BRAKE + WARNING": 0.3}.get(decision, 0.0)
    if state in {"drowsy", "unavailable"}:
        level = max(level, 0.8)      # controlled stop when monitoring cannot confirm an alert driver
    return level

def head_controls(snap):
    """Return throttle, steering and brake; missing tracking commands a stop."""
    if not snap["face"] or not snap["calibrated"] or snap.get("state") in {"drowsy", "unavailable"}:
        return 0.0, 0.0, 0.8
    yaw, pitch = snap["yaw"], snap["pitch"]
    steer = math.copysign(min(1.0, max(0.0, abs(yaw) - HEAD_STEER_DEADZONE) /
                             (HEAD_STEER_FULL - HEAD_STEER_DEADZONE)), yaw)
    pedal = min(1.0, max(0.0, abs(pitch) - HEAD_PEDAL_DEADZONE) /
                (HEAD_PEDAL_FULL - HEAD_PEDAL_DEADZONE))
    return (0.4 * pedal if pitch > 0 else 0.0), HEAD_MAX_STEER * steer, (0.8 * pedal if pitch < 0 else 0.0)

class HeadController:
    """Limit changes in steering/throttle; braking always takes effect immediately."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.throttle = self.steer = 0.0
        self.updated = time.monotonic()

    def update(self, snap):
        throttle, steer, brake = head_controls(snap)
        now = time.monotonic()
        dt = max(0.0, min(now - self.updated, 0.1))
        self.updated = now
        if not snap["face"] or not snap["calibrated"] or snap.get("state") in {"drowsy", "unavailable"}:
            self.reset()
            return 0.0, 0.0, brake
        def approach(current, target, rate):
            return current + max(-rate * dt, min(rate * dt, target - current))
        self.steer = approach(self.steer, steer, HEAD_STEER_RATE)
        self.throttle = 0.0 if brake else approach(self.throttle, throttle, HEAD_THROTTLE_RATE)
        return self.throttle, self.steer, brake


def control_action(state, mode, controls):
    if mode == "EVALUATION_ONLY":
        return "EVALUATION ONLY"
    if controls is None:
        return "AUTOPILOT"
    if controls[2] > 0:
        return {"drowsy": "DROWSY STOP", "unavailable": "TRACKING LOST STOP"}.get(
            state, "LOOK STRAIGHT TO DRIVE" if mode == "LOOK_STRAIGHT_WAIT" else "BRAKE")
    return "DRIVER ALERT" if state == "distracted" else "DRIVE"


def log_scenario(sample, mode, controls, action, snap=None):
    """Log commanded controls, not measured actuator response."""
    speed, distance, obstacle, road, state, ttc, risk, decision = sample
    new = not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["time", "speed_kmh", "distance_m", "obstacle", "road",
                             "driver_state", "ttc_s", "risk", "risk_decision", "mode",
                             "control_action", "commanded_throttle", "commanded_steer", "commanded_brake",
                             "yaw_deg", "pitch_deg", "gaze_yaw_deg", "ear", "ear_threshold"])
        writer.writerow([datetime.now().isoformat(timespec="milliseconds"), round(speed, 1),
                         round(distance, 1) if math.isfinite(distance) else "NO_LEAD_IN_RANGE",
                         obstacle if math.isfinite(distance) else "None detected", road, state, round(ttc, 2) if math.isfinite(ttc) else "N/A",
                         risk, decision, mode, action,
                         *(tuple(round(v, 3) for v in controls) if controls is not None else ("", "", "")),
                         *((snap or {}).get(name, "") if (snap or {}).get("face") else ""
                           for name in ("yaw", "pitch", "gaze_yaw", "ear", "ear_threshold"))])

def listen_once():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source, timeout=5, phrase_time_limit=6)
        return r.recognize_google(audio).lower()
    except Exception as e:
        print("Voice error:", e)
        return ""

def parse_command(text):
    text, out = text.lower(), {}
    for field in ("speed", "distance"):
        match = re.search(rf"{field}\D{{0,15}}?(\d+(?:\.\d+)?)", text)
        if match:
            out[field] = match.group(1)
    for field, words in [("obstacle", ("vehicle", "animal", "pothole")),
                         ("road", ("dry", "wet", "poor"))]:
        for word in words:
            if re.search(rf"\b{word}\b", text):
                out[field] = word.capitalize()
    return out

def connect_carla(host, port, timeout=45.0):
    """Contact the configured server directly, with a bounded retry period."""
    client = carla.Client(host, port)
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            client.set_timeout(max(0.01, min(2.0, deadline - time.monotonic())))
            world = client.get_world()
            client.set_timeout(2.0)
            return client, world
        except RuntimeError as error:
            last_error = error
            time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
    raise RuntimeError(f"CARLA unavailable at {host}:{port}: {last_error}")

class CarlaIntegration:
    def __init__(self):
        self.vehicle = None
        self.camera = None
        self.npcs = []
        self.autopilot = False
        self._road_frame = None
        self._lock = threading.Lock()
        self._brake_until = 0.0
        self._held_brake = 0.0
        self._lead_cache = (0.0, math.inf)

        if carla is None:
            print("carla module not installed - local emulation mode.")
            return
        try:
            host = os.environ.get("CARLA_HOST", "127.0.0.1")
            port = int(os.environ.get("CARLA_PORT", "2000"))
            self.client, self.world = connect_carla(host, port)
            self.spectator = self.world.get_spectator()
            bp_lib = self.world.get_blueprint_library()

            spawn_points = self.world.get_map().get_spawn_points()
            random.shuffle(spawn_points)
            model3 = bp_lib.filter("model3")
            ego_bp = model3[0]
            for i, sp in enumerate(spawn_points):
                self.vehicle = self.world.try_spawn_actor(ego_bp, sp)
                if self.vehicle:
                    spawn_points = spawn_points[i + 1:]
                    break
            if self.vehicle is None:
                raise RuntimeError("no free spawn point for the ego vehicle")

            # forward road camera (fixed to the car) -> feeds YOLO
            cam_bp = bp_lib.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", "640")
            cam_bp.set_attribute("image_size_y", "480")
            self.camera = self.world.spawn_actor(
                cam_bp, carla.Transform(carla.Location(x=1.6, z=1.5)), attach_to=self.vehicle)
            self.camera.listen(self._on_image)

            # some traffic so there's something to react to
            car_bps = [b for b in bp_lib.filter("vehicle.*")
                       if int(b.get_attribute("number_of_wheels")) == 4]
            for sp in spawn_points:
                if len(self.npcs) >= NUM_NPC:
                    break
                npc = self.world.try_spawn_actor(random.choice(car_bps), sp)
                if npc:
                    self.npcs.append(npc)
                    npc.set_autopilot(True)

            self.vehicle.set_autopilot(False)
            self.autopilot = False
            print(f"CARLA connected: ego vehicle + {len(self.npcs)} NPC vehicles.")
        except Exception as e:
            print(f"CARLA connection failed: {e}. Running in local emulation mode.")
            try:
                wx.MessageBox(str(e), "CARLA")
            except Exception:
                pass
            self.cleanup()
            self.vehicle = None

    def _on_image(self, image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        with self._lock:
            self._road_frame = arr[:, :, :3].copy()      # BGRA -> BGR

    def get_road_frame(self):
        with self._lock:
            return None if self._road_frame is None else self._road_frame.copy()

    def update_view(self):
        """Keep the spectator above/behind the trunk; head motion never rotates it."""
        if not self.vehicle:
            return
        tf = self.vehicle.get_transform()
        f, r, u = tf.get_forward_vector(), tf.get_right_vector(), tf.get_up_vector()
        loc = tf.location
        eye = carla.Location(x=loc.x + f.x * CHASE_X + r.x * CHASE_Y + u.x * CHASE_Z,
                             y=loc.y + f.y * CHASE_X + r.y * CHASE_Y + u.y * CHASE_Z,
                             z=loc.z + f.z * CHASE_X + r.z * CHASE_Y + u.z * CHASE_Z)
        rot = carla.Rotation(pitch=tf.rotation.pitch + CHASE_PITCH, yaw=tf.rotation.yaw)
        self.spectator.set_transform(carla.Transform(eye, rot))

    def set_autopilot(self, enabled):
        if not self.vehicle or self.autopilot == enabled:
            return
        self.vehicle.set_autopilot(enabled)
        self.autopilot = enabled
        if not enabled:
            self.vehicle.apply_control(carla.VehicleControl())

    def apply_manual_control(self, throttle=0.0, steer=0.0, brake=0.0):
        if not self.vehicle:
            return
        self.set_autopilot(False)
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=0.0 if brake > 0 else max(0.0, min(1.0, throttle)),
            steer=max(-1.0, min(1.0, steer)),
            brake=max(0.0, min(1.0, brake))))

    def get_speed(self):
        v = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

    def get_lead_distance(self, max_range=80.0):
        """Bumper-to-bumper distance to the nearest vehicle ahead in our lane (cached 0.1 s)."""
        now = time.time()
        if now - self._lead_cache[0] < 0.1:
            return self._lead_cache[1]
        tf = self.vehicle.get_transform()
        f, r, loc = tf.get_forward_vector(), tf.get_right_vector(), tf.location
        best = max_range
        for npc in self.world.get_actors().filter("vehicle.*"):
            if npc.id == self.vehicle.id:
                continue
            try:
                p = npc.get_location()
            except Exception:
                continue
            dx, dy = p.x - loc.x, p.y - loc.y
            along = dx * f.x + dy * f.y
            lateral = dx * r.x + dy * r.y
            if 0 < along < best and abs(lateral) < LANE_HALF_WIDTH:
                best = along
        dist = max(best - CAR_LENGTH_M, 0.5) if best < max_range else math.inf
        self._lead_cache = (now, dist)
        return dist

    def set_brake(self, level):
        """level > 0: take over from autopilot and brake. level == 0: hand back after a hold time."""
        if not self.vehicle:
            return
        now = time.time()
        if level > 0:
            self.set_autopilot(False)    # Traffic Manager would overwrite our control
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=level))
            self._brake_until = now + BRAKE_HOLD_S
            self._held_brake = level
        elif not self.autopilot and now > self._brake_until:
            self.vehicle.apply_control(carla.VehicleControl(brake=0.0))
            self.set_autopilot(True)
        return None if self.autopilot else (0.0, 0.0, self._held_brake)

    def cleanup(self):
        for actor in ([self.camera] if self.camera else []) + self.npcs + ([self.vehicle] if self.vehicle else []):
            try:
                if actor is self.camera:
                    actor.stop()
                actor.destroy()
            except Exception:
                pass
        self.camera, self.npcs, self.vehicle = None, [], None
        print("CARLA actors cleaned up.")

class RoadVision(FrameWorker):
    def __init__(self, carla_sim, model, device):
        super().__init__()
        self.carla_sim, self.model, self.device = carla_sim, model, device

    def run(self):
        while not self._stop_evt.is_set():
            frame = self.carla_sim.get_road_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            try:
                res = self.model(frame, device=self.device, verbose=False)
                with self._lock:
                    self._frame = res[0].plot()
            except Exception as e:
                print("YOLO error:", e)
                time.sleep(1)

class ADASApp(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Unified ADAS System", size=(620, 880))

        self.carla_sim = CarlaIntegration()
        self.tracker = DriverTracker()
        self.tracker.start()

        self.vision = None
        if ENABLE_YOLO and self.carla_sim.vehicle:
            try:
                device = 0 if torch.cuda.is_available() else "cpu"
                self.vision = RoadVision(self.carla_sim, YOLO("yolov10n.pt"), device)
                self.vision.start()
            except Exception as e:
                print(f"YOLO disabled: {e}")

        self.head_controller = HeadController()
        self.straight_gate = StraightDriveGate()
        self._driver_snap = None
        self._evaluation = None
        self._last_log = 0.0
        self._last_beep = 0.0
        self._last_key = None
        self._camera_warned = False
        self._closing = False

        panel = wx.ScrolledWindow(self)
        panel.SetScrollRate(0, 10)
        layout = wx.BoxSizer(wx.VERTICAL)

        def add(widget):
            layout.Add(widget, 0, wx.EXPAND | wx.ALL, 5)
            return widget

        def label(text):
            return add(wx.StaticText(panel, label=text))

        def button(text, handler):
            widget = add(wx.Button(panel, label=text))
            widget.Bind(wx.EVT_BUTTON, handler)
            return widget

        label("ADAS Risk Assessment Tool")
        self.driver_label = label("Driver State: starting camera...")
        self.head_label = label("Head: --")
        button("Re-center head (look straight first)", lambda e: self.tracker.recalibrate())
        connected = self.carla_sim.vehicle is not None
        for name, text, value, handler in [
            ("auto_chk", "Use live CARLA speed + lead distance", connected, lambda e: self._apply_auto_state()),
            ("straight_chk", "Drive only while looking straight (autopilot steers)", False, self.on_straight_toggle),
            ("head_control_chk", "Drive with head movement (enable after calibration)", False, self.on_head_control_toggle),
            ("autopilot_chk", "CARLA autopilot", False, self.on_autopilot_toggle),
        ]:
            widget = add(wx.CheckBox(panel, label=text))
            widget.SetValue(value)
            widget.Enable(connected)
            widget.Bind(wx.EVT_CHECKBOX, handler)
            setattr(self, name, widget)
        label("Look-straight mode: face forward to drive; look away to stop.")
        label("Head: yaw steers; look up to accelerate, down to brake. Keyboard: WASD/arrows.")
        self.command_input = add(wx.TextCtrl(panel, style=wx.TE_PROCESS_ENTER))
        self.command_input.SetHint("Set speed at 80 and distance 20")
        self.command_input.Bind(wx.EVT_TEXT_ENTER, self.on_command)
        for name, text, value in [
            ("speed_input", "Speed (km/h)", "0"),
            ("distance_input", "Distance (m)", "30"),
            ("obstacle_choice", "Obstacle", ["Vehicle", "Animal", "Pothole"]),
            ("road_choice", "Road", ["Dry", "Wet", "Poor"]),
        ]:
            label(text)
            if isinstance(value, list):
                widget = wx.Choice(panel, choices=value)
                widget.SetSelection(0)
            else:
                widget = wx.TextCtrl(panel, value=value)
            setattr(self, name, add(widget))
        button("Evaluate ADAS (log)", self.on_evaluate)
        for name, text in [("ttc_output", "TTC: --"), ("risk_output", "Risk: --"),
                           ("decision_output", "Decision: --")]:
            setattr(self, name, label(text))
        self.control_output = label("Control: --")
        self.voice_btn = button("Voice Command", self.on_voice_input)
        panel.SetSizer(layout)
        self._apply_auto_state()
        self.Bind(wx.EVT_CLOSE, self.on_close)

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_tick, self.timer)
        self.timer.Start(50)       # 20 Hz control loop
        self.Show()

    def _is_auto(self):
        return self.auto_chk.GetValue() and self.carla_sim.vehicle is not None

    def _apply_auto_state(self):
        manual = not self._is_auto()
        self.speed_input.Enable(manual)
        self.distance_input.Enable(manual)

    def on_autopilot_toggle(self, _event):
        self.head_controller.reset()
        enabled = self.autopilot_chk.GetValue()
        if enabled:
            self.straight_chk.SetValue(False)
            self.straight_gate.reset()
            self.head_control_chk.SetValue(False)
        self.carla_sim.set_autopilot(enabled)

    def on_head_control_toggle(self, _event):
        self.head_controller.reset()
        if self.head_control_chk.GetValue():
            self.straight_chk.SetValue(False)
            self.straight_gate.reset()
            self.autopilot_chk.SetValue(False)
            self.carla_sim.set_autopilot(False)

    def on_straight_toggle(self, _event):
        self.straight_gate.reset()
        self.head_controller.reset()
        if self.straight_chk.GetValue():
            self.head_control_chk.SetValue(False)
            self.autopilot_chk.SetValue(False)
        self.carla_sim.set_autopilot(False)

    @staticmethod
    def _key_down(vk_code):
        try:
            windows_code = {wx.WXK_UP: 0x26, wx.WXK_DOWN: 0x28, wx.WXK_LEFT: 0x25, wx.WXK_RIGHT: 0x27}.get(vk_code, vk_code)
            return bool(ctypes.windll.user32.GetAsyncKeyState(windows_code) & 0x8000)
        except (AttributeError, OSError):
            return bool(wx.GetKeyState(vk_code))

    def _manual_keys(self):
        if isinstance(wx.Window.FindFocus(), wx.TextCtrl):
            return 0.0, 0.0, 0.0
        forward = self._key_down(ord("W")) or self._key_down(wx.WXK_UP)  # W / Up
        back = self._key_down(ord("S")) or self._key_down(wx.WXK_DOWN)     # S / Down
        left = self._key_down(ord("A")) or self._key_down(wx.WXK_LEFT)     # A / Left
        right = self._key_down(ord("D")) or self._key_down(wx.WXK_RIGHT)    # D / Right
        throttle = 0.65 if forward else 0.0
        brake = 0.7 if back else 0.0
        steer = float(right) - float(left)
        return throttle, steer, brake

    def _set(self, ctrl, text):
        if ctrl.GetLabel() != text:
            ctrl.SetLabel(text)

    def _read_inputs(self):
        if self._is_auto():
            sp, di = self.carla_sim.get_speed(), self.carla_sim.get_lead_distance()
            self.speed_input.ChangeValue(f"{sp:.1f}")
            self.distance_input.ChangeValue(f"{di:.1f}" if math.isfinite(di) else "Clear")
            return sp, di
        try:
            sp, di = float(self.speed_input.GetValue()), float(self.distance_input.GetValue())
        except ValueError:
            return None
        return (sp, di) if math.isfinite(sp) and sp >= 0 and math.isfinite(di) and di > 0 else None

    def on_tick(self, _event):
        if self.tracker.error:
            self._set(self.driver_label, f"Driver State: unavailable ({self.tracker.error})")
            if not self._camera_warned:
                self._camera_warned = True
                wx.MessageBox(f"{self.tracker.error}. Driver monitoring is off.", "Camera")
            snap = dict(state="unavailable", face=False, yaw=0.0, pitch=0.0, gaze_yaw=0.0, calibrated=False)
        else:
            snap = self.tracker.snapshot()
        state = snap["state"]
        self._driver_snap = snap

        self.carla_sim.update_view()

        decision = self.evaluate(state)

        adas_brake = brake_level(decision, state, self._is_auto())
        if self.straight_chk.GetValue():
            allowed = self.straight_gate.allow(snap)
            mode = "LOOK_STRAIGHT" if allowed else "LOOK_STRAIGHT_WAIT"
            controls = self.carla_sim.set_brake(adas_brake if allowed else max(0.6, adas_brake))
        elif self.autopilot_chk.GetValue():
            mode = "AUTOPILOT"
            controls = self.carla_sim.set_brake(adas_brake)
        else:
            mode = "HEAD" if self.head_control_chk.GetValue() else "KEYBOARD"
            if mode == "HEAD":
                throttle, steer, brake = self.head_controller.update(snap)
            else:
                throttle, steer, brake = self._manual_keys()
            brake = max(brake, adas_brake)
            controls = (0.0 if brake > 0 else throttle), steer, brake
            self.carla_sim.apply_manual_control(*controls)
        if self.carla_sim.vehicle is None:
            mode, controls = "EVALUATION_ONLY", None
        self.record_controls(mode, controls)
        now = time.time()
        alert = {"drowsy": (1.0, 1000, 300), "distracted": (2.0, 600, 200)}.get(state)
        if alert and now - self._last_beep > alert[0]:
            beep_async(alert[1], alert[2])
            self._last_beep = now

        self._set(self.driver_label, f"Driver State: {self.tracker.error or state.upper()}")
        if snap["face"] and snap["calibrated"]:
            self._set(self.head_label, f"Head yaw {snap['yaw']:+.0f}°  pitch {snap['pitch']:+.0f}°  gaze {snap['gaze_yaw']:+.0f}°")
        elif not snap["calibrated"]:
            self._set(self.head_label, "Calibrating - look straight at the screen")
        else:
            self._set(self.head_label, "No face detected")

        frame = self.tracker.get_frame()
        if frame is not None:
            if state == "drowsy":
                cv2.putText(frame, "WAKE UP! DROWSY DRIVER!", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            elif state == "distracted":
                cv2.putText(frame, "EYES ON THE ROAD!", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 3)
            cv2.imshow("Driver Monitoring", frame)
        if self.vision:
            road = self.vision.get_frame()
            if road is not None:
                cv2.imshow("Road View (YOLO)", road)
        cv2.waitKey(1)

    def evaluate(self, state, force_log=False):
        inputs = self._read_inputs()
        if inputs is None:
            self._evaluation = None
            self._set(self.decision_output, "Decision: invalid input")
            return None
        speed, distance = inputs
        obstacle = self.obstacle_choice.GetStringSelection()
        road = self.road_choice.GetStringSelection()
        ttc, risk, decision = evaluate_adas(speed, distance, obstacle, road, state)
        for control, text in [(self.ttc_output, f"TTC: {ttc} seconds" if math.isfinite(ttc) else "TTC: N/A (stopped or no lead)"),
                              (self.risk_output, f"Risk Score: {risk}"),
                              (self.decision_output, f"Decision: {decision}")]:
            self._set(control, text)
        self._evaluation = (speed, distance, obstacle, road, state, ttc, risk, decision)
        if force_log:
            self.record_controls("EVALUATION_ONLY", None, force=True)
        return decision

    def record_controls(self, mode, controls, force=False):
        if self._evaluation is None:
            return
        state, decision = self._evaluation[4], self._evaluation[7]
        action = control_action(state, mode, controls)
        self._set(self.control_output, f"Control: {action}" +
                  (f" | brake {controls[2]:.0%}" if controls is not None else ""))
        key = state, decision, mode, action
        now = time.monotonic()
        if force or key != self._last_key or now - self._last_log >= 1.0:
            log_scenario(self._evaluation, mode, controls, action, self._driver_snap)
            self._last_key, self._last_log = key, now

    def on_evaluate(self, _event):
        self._driver_snap = self.tracker.snapshot()
        if self.evaluate(self._driver_snap["state"], force_log=True) is None:
            wx.MessageBox("Enter finite numbers: speed >= 0, distance > 0", "Input Error")

    def on_command(self, _event):
        parsed = parse_command(self.command_input.GetValue().lower())
        if "speed" in parsed or "distance" in parsed:
            self.auto_chk.SetValue(False)          # typed numbers mean "what-if" mode
            self._apply_auto_state()
        for name in ("speed", "distance", "obstacle", "road"):
            if name in parsed:
                control = getattr(self, name + ("_input" if name in {"speed", "distance"} else "_choice"))
                setter = control.SetValue if name in {"speed", "distance"} else control.SetStringSelection
                setter(parsed[name])
        self.on_evaluate(None)

    def on_voice_input(self, _event):
        self.voice_btn.Disable()
        self.voice_btn.SetLabel("Listening...")
        threading.Thread(target=lambda: wx.CallAfter(self._voice_done, listen_once()), daemon=True).start()

    def _voice_done(self, text):
        if self._closing:
            return
        self.voice_btn.Enable()
        self.voice_btn.SetLabel("Voice Command")
        if text:
            self.command_input.SetValue(text)
            self.on_command(None)
        else:
            wx.MessageBox("Didn't catch that (check mic / internet).", "Voice")

    def on_close(self, _event):
        self._closing = True
        self.timer.Stop()
        self.tracker.stop()
        if self.vision:
            self.vision.stop()
        self.carla_sim.cleanup()
        cv2.destroyAllWindows()
        self.Destroy()

if __name__ == "__main__":
    app = wx.App()
    ADASApp()
    app.MainLoop()
