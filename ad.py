#=====================================================================
# 1. IMPORT ORDER MATTERS ON WINDOWS: torch + ultralytics FIRST
# =====================================================================

import torch
from ultralytics import YOLO
 
import csv
import ctypes
import json
import math
import os
import random
import re
import subprocess
import sys
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


def _agent_log(hypothesis_id, location, message, data=None):
    try:
        with open(_AGENT_LOG, "a", encoding="utf-8") as _f:
            _f.write(json.dumps({
                "sessionId": "251a39",
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data or {},
                "timestamp": int(time.time() * 1000),
            }) + "\n")
    except Exception:
        pass

_agent_log("A", "ad.py:import", "carla import result", {
    "carla_is_none": carla is None,
    "executable": sys.executable,
    "version": sys.version,
    "carla_file": getattr(carla, "__file__", None),
})

 
# =====================================================================
# 2. TUNABLE SETTINGS
# =====================================================================
LOG_FILE = "adas_log.csv"
 
# --- Driver monitoring ---
EAR_THRESHOLD = 0.21            # eyes "closed" below this
DROWSY_SECONDS = 2.0            # eyes closed this long -> drowsy
DISTRACT_SECONDS = 2.0          # eyes off road this long -> distracted
OFF_ROAD_YAW_DEG = 25.0         # head+gaze turned further than this = off road
OFF_ROAD_PITCH_DOWN_DEG = 20.0  # head tilted down further than this = off road (phone)
CALIBRATION_FRAMES = 30         # ~1 s of "look straight at the screen"
GAZE_DEG_PER_UNIT = 100.0       # iris offset (fraction of eye width) -> degrees
 
# --- Head/eye -> CARLA camera ---
YAW_GAIN = 2.0                  # 20 deg real head turn -> 40 deg in sim
PITCH_GAIN = 1.5
GAZE_VIEW_WEIGHT = 0.3          # how much eye movement nudges the view (0 = head only)
MAX_VIEW_YAW = 120.0
MAX_VIEW_PITCH = 50.0
VIEW_SMOOTHING = 0.35           # 0..1, higher = snappier, lower = smoother
EYE_X, EYE_Y, EYE_Z = 0.30, -0.35, 1.42   # driver's eye in car frame (x fwd, y right, z up)
HEAD_STEER_DEADZONE = 4.0       # degrees from calibrated neutral
HEAD_STEER_FULL = 25.0          # degrees for full steering input
HEAD_PEDAL_DEADZONE = 8.0       # pitch degrees before throttle/brake responds
HEAD_PEDAL_FULL = 25.0          # pitch degrees for full pedal input
 
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
 
 
# =====================================================================
# 3. DRIVER TRACKING (MediaPipe, own thread): eyes + head pose + gaze
# =====================================================================
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
 
 
class DriverTracker(threading.Thread):
    def __init__(self, camera_index=0):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.error = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._snap = dict(state="normal", face=False, yaw=0.0, pitch=0.0,
                          gaze_yaw=0.0, ear=None, calibrated=False)
        self._frame = None
        self._closed_since = None
        self._away_since = None
        self._cal = []
        self._neutral = None
 
    def recalibrate(self):
        self._cal = []
        self._neutral = None
 
    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2)
 
    def snapshot(self):
        with self._lock:
            return dict(self._snap)
 
    def get_frame(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()
 
    def run(self):
        try:
            self._run_tracker()
        except Exception as e:
            self.error = f"Driver tracking failed: {type(e).__name__}: {e}"
            print(self.error)
            traceback.print_exc()
            cap = getattr(self, "_cap", None)
            if cap is not None:
                cap.release()

    def _run_tracker(self):
        cap = cv2.VideoCapture(self.camera_index)
        self._cap = cap
        if not cap.isOpened():
            self.error = f"Could not open camera {self.camera_index}"
            cap.release()
            return
 
        with mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5) as mesh:
            while not self._stop_evt.is_set():
                ok, raw = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue
                h, w = raw.shape[:2]
                res = mesh.process(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))   # un-mirrored frame
                now = time.time()
 
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
                            self._cal.append((raw_yaw, raw_pitch, gaze_x))
                            if len(self._cal) >= CALIBRATION_FRAMES:
                                self._neutral = tuple(np.mean(self._cal, axis=0))
                        n = self._neutral
                        if n is not None:
                            yaw, pitch = raw_yaw - n[0], raw_pitch - n[1]
                            gaze_yaw = (gaze_x - n[2]) * GAZE_DEG_PER_UNIT
 
                calibrated = self._neutral is not None
                # EAR is unreliable when the head is turned far, so ignore it then
                closed = face and ear < EAR_THRESHOLD and abs(yaw) < 40
                off_road = (not face) or (calibrated and (
                    abs(yaw + gaze_yaw) > OFF_ROAD_YAW_DEG or pitch < -OFF_ROAD_PITCH_DOWN_DEG))
 
                self._closed_since = (self._closed_since or now) if closed else None
                self._away_since = (self._away_since or now) if off_road else None
 
                if self._closed_since and now - self._closed_since >= DROWSY_SECONDS:
                    state = "drowsy"
                elif self._away_since and now - self._away_since >= DISTRACT_SECONDS:
                    state = "distracted"
                else:
                    state = "normal"
 
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
                                      gaze_yaw=gaze_yaw, ear=ear, calibrated=calibrated)
                    self._frame = disp
        cap.release()
 
 
# =====================================================================
# 4. ADAS LOGIC, LOGGING, COMMANDS
# =====================================================================
def evaluate_adas(speed_kmh, distance_m, obstacle, road, driver_state):
    speed_ms = speed_kmh / 3.6
    ttc = distance_m / speed_ms if speed_ms > 0 else 999
    road_factor = {"Dry": 1.0, "Wet": 1.5, "Poor": 2.0}.get(road, 1.0)
    risk = (speed_ms * road_factor) / ttc
    if obstacle == "Animal":
        risk *= 1.3
    elif obstacle == "Pothole":
        risk *= 0.8
    risk += {"normal": 0, "drowsy": 7, "distracted": 4}.get(driver_state, 0)
 
    if risk > 25:
        decision = "EMERGENCY BRAKE"
    elif risk > 15:
        decision = "GRADUAL BRAKE + ALERT"
    elif risk > 8:
        decision = "PRE-BRAKE + WARNING"
    else:
        decision = "MONITOR"
    return round(ttc, 2), round(risk, 2), decision
 
 
def brake_level(decision, state, auto):
    """Auto mode: obey the ADAS decision. Manual what-if mode: only react to the driver."""
    level = 0.0
    if auto:
        level = {"EMERGENCY BRAKE": 1.0, "GRADUAL BRAKE + ALERT": 0.6,
                 "PRE-BRAKE + WARNING": 0.3}.get(decision, 0.0)
    if state == "drowsy":
        level = max(level, 0.8)      # controlled stop for a sleeping driver
    return level
 
 
def log_scenario(speed, distance, obstacle, road, driver_state, ttc, risk, decision):
    new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "speed_kmh", "distance_m", "obstacle", "road",
                        "driver_state", "ttc_s", "risk", "decision"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    round(speed, 1), round(distance, 1), obstacle, road,
                    driver_state, ttc, risk, decision])
 
 
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
    out = {}
    m = re.search(r"speed\D{0,15}?(\d+(?:\.\d+)?)", text)
    if m:
        out["speed"] = m.group(1)
    m = re.search(r"distance\D{0,15}?(\d+(?:\.\d+)?)", text)
    if m:
        out["distance"] = m.group(1)
    for word in ("vehicle", "animal", "pothole"):
        if word in text:
            out["obstacle"] = word.capitalize()
    for word in ("dry", "wet", "poor"):
        if re.search(rf"\b{word}\b", text):
            out["road"] = word.capitalize()
    return out
 
 
# =====================================================================
# 5. CARLA INTEGRATION
# =====================================================================
def _pids_listening(port):
    """PIDs with a TCP LISTENING socket on port (Windows netstat)."""
    pids = set()
    try:
        kwargs = {"text": True, "errors": "ignore"}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        out = subprocess.check_output(["netstat", "-ano", "-p", "tcp"], **kwargs)
        suffix = ":" + str(port)
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 5 or parts[0].upper() != "TCP":
                continue
            if parts[3].upper() != "LISTENING":
                continue
            if parts[1].endswith(suffix):
                pids.add(int(parts[-1]))
    except Exception:
        pass
    return pids


def _wait_for_carla_world(host, port, deadline_s=45.0):
    """Handshake like CARLA's test_connection.py, then fetch the world.

    Two CarlaUE4 processes bound to the same RPC port accept TCP but never
    answer get_world(); fail fast in that case instead of hanging.
    """
    pids = _pids_listening(port)
    # #region agent log
    _agent_log("F", "ad.py:_wait_for_carla_world", "port listeners", {
        "host": host, "port": port, "pids": sorted(pids),
    })
    # #endregion
    if not pids:
        raise RuntimeError(
            f"CARLA is not listening on {host}:{port}. Start CarlaUE4.exe and wait until the town is visible."
        )
    if len(pids) > 1:
        raise RuntimeError(
            f"CARLA RPC port {port} is claimed by {len(pids)} processes (PIDs {sorted(pids)}). "
            "A leftover CarlaUE4 instance is stealing the connection. "
            "In Task Manager, end extra CarlaUE4 / CarlaUE4-Win64-Shipping tasks so only ONE simulator is running, "
            "then start this app again."
        )

    client = carla.Client(host, port)
    t0 = time.time()
    last_err = None
    while time.time() - t0 < deadline_s:
        try:
            client.set_timeout(2.0)
            server_ver = client.get_server_version()
            client.set_timeout(20.0)
            world = client.get_world()
            # #region agent log
            _agent_log("F", "ad.py:_wait_for_carla_world", "handshake ok", {
                "server_version": str(server_ver), "elapsed_s": round(time.time() - t0, 2),
            })
            # #endregion
            return client, world, server_ver
        except RuntimeError as e:
            last_err = e
            time.sleep(0.5)
    raise RuntimeError(
        f"{last_err}. The simulator on port {port} (PID {next(iter(pids))}) did not become ready. "
        "Wait until the CARLA town is fully loaded, or restart CarlaUE4.exe once."
    )


class CarlaIntegration:
    def __init__(self):
        self.vehicle = None
        self.camera = None
        self.npcs = []
        self.autopilot = False
        self._road_frame = None
        self._lock = threading.Lock()
        self._brake_until = 0.0
        self._lead_cache = (0.0, 100.0)
 
        if carla is None:
            print("carla module not installed - local emulation mode.")
            # #region agent log
            _agent_log("A", "ad.py:CarlaIntegration.__init__", "skip connect: carla module missing", {
                "executable": sys.executable,
            })
            # #endregion
            return
        try:
            host = os.environ.get("CARLA_HOST", "127.0.0.1")
            port = int(os.environ.get("CARLA_PORT", "2000"))
            # #region agent log
            _agent_log("B", "ad.py:CarlaIntegration.__init__", "connecting to CARLA", {
                "host": host, "port": port, "executable": sys.executable,
                "client_version": str(carla.Client(host, port).get_client_version()),
            })
            # #endregion
            self.client, self.world, server_ver = _wait_for_carla_world(host, port)
            self.spectator = self.world.get_spectator()
            bp_lib = self.world.get_blueprint_library()
            # #region agent log
            _agent_log("B", "ad.py:CarlaIntegration.__init__", "get_world succeeded", {
                "map": self.world.get_map().name,
                "server_version": str(server_ver),
            })
            # #endregion
 
            spawn_points = self.world.get_map().get_spawn_points()
            random.shuffle(spawn_points)
            model3 = bp_lib.filter("model3")
            # #region agent log
            _agent_log("C", "ad.py:CarlaIntegration.__init__", "spawn prep", {
                "spawn_count": len(spawn_points),
                "model3_count": len(model3),
                "model3_ids": [b.id for b in model3],
            })
            # #endregion
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
            # #region agent log
            _agent_log("D", "ad.py:CarlaIntegration.__init__", "ego+camera spawned", {
                "vehicle_id": self.vehicle.id if self.vehicle else None,
            })
            # #endregion
 
            # some traffic so there's something to react to
            car_bps = [b for b in bp_lib.filter("vehicle.*")
                       if int(b.get_attribute("number_of_wheels")) == 4]
            for sp in spawn_points:
                if len(self.npcs) >= NUM_NPC:
                    break
                npc = self.world.try_spawn_actor(random.choice(car_bps), sp)
                if npc:
                    npc.set_autopilot(True)
                    self.npcs.append(npc)
 
            self.vehicle.set_autopilot(False)
            self.autopilot = False
            print(f"CARLA connected: ego vehicle + {len(self.npcs)} NPC vehicles.")
            # #region agent log
            _agent_log("E", "ad.py:CarlaIntegration.__init__", "CARLA setup complete", {
                "npc_count": len(self.npcs), "autopilot": self.autopilot,
            })
            # #endregion
        except Exception as e:
            print(f"CARLA connection failed: {e}. Running in local emulation mode.")
            # #region agent log
            _agent_log("B", "ad.py:CarlaIntegration.__init__", "CARLA connection failed", {
                "error_type": type(e).__name__,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
            # #endregion
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
 
    def update_view(self, yaw_off, pitch_off):
        """Put the spectator at the driver's eyes and rotate it with the driver's head."""
        if not self.vehicle:
            return
        tf = self.vehicle.get_transform()
        f, r, u = tf.get_forward_vector(), tf.get_right_vector(), tf.get_up_vector()
        loc = tf.location
        eye = carla.Location(
            x=loc.x + f.x * EYE_X + r.x * EYE_Y + u.x * EYE_Z,
            y=loc.y + f.y * EYE_X + r.y * EYE_Y + u.y * EYE_Z,
            z=loc.z + f.z * EYE_X + r.z * EYE_Y + u.z * EYE_Z)
        rot = carla.Rotation(pitch=tf.rotation.pitch + pitch_off,
                             yaw=tf.rotation.yaw + yaw_off,
                             roll=tf.rotation.roll)
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
            throttle=max(0.0, min(1.0, throttle)),
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
        for npc in self.npcs:
            try:
                p = npc.get_location()
            except Exception:
                continue
            dx, dy = p.x - loc.x, p.y - loc.y
            along = dx * f.x + dy * f.y
            lateral = dx * r.x + dy * r.y
            if 0 < along < best and abs(lateral) < LANE_HALF_WIDTH:
                best = along
        dist = max(best - CAR_LENGTH_M, 0.5) if best < max_range else max_range
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
        elif not self.autopilot and now > self._brake_until:
            self.vehicle.apply_control(carla.VehicleControl(brake=0.0))
            self.set_autopilot(True)
 
    def cleanup(self):
        for actor in ([self.camera] if self.camera else []) + self.npcs + ([self.vehicle] if self.vehicle else []):
            try:
                if actor is self.camera:
                    actor.stop()
                actor.destroy()
            except Exception:
                pass
        self.camera, self.npcs = None, []
        print("CARLA actors cleaned up.")
 
 
# =====================================================================
# 6. YOLO ON THE CARLA ROAD CAMERA (own thread so the GUI never stalls)
# =====================================================================
class RoadVision(threading.Thread):
    def __init__(self, carla_sim, model, device):
        super().__init__(daemon=True)
        self.carla_sim, self.model, self.device = carla_sim, model, device
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._annotated = None
 
    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2)
 
    def get_frame(self):
        with self._lock:
            return None if self._annotated is None else self._annotated.copy()
 
    def run(self):
        while not self._stop_evt.is_set():
            frame = self.carla_sim.get_road_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            try:
                res = self.model(frame, device=self.device, verbose=False)
                with self._lock:
                    self._annotated = res[0].plot()
            except Exception as e:
                print("YOLO error:", e)
                time.sleep(1)
 
 
# =====================================================================
# 7. GUI
# =====================================================================
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
 
        self.view_yaw = 0.0
        self.view_pitch = 0.0
        self._last_beep = 0.0
        self._last_key = None
        self._camera_warned = False
 
        panel = wx.Panel(self)
        panel.SetBackgroundColour("Beige")
        bold = lambda sz: wx.Font(sz, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
 
        title = wx.StaticText(panel, label="ADAS Risk Assessment Tool")
        title.SetFont(bold(16))
        self.driver_label = wx.StaticText(panel, label="Driver State: starting camera...")
        self.driver_label.SetFont(bold(11))
        self.head_label = wx.StaticText(panel, label="Head: --")
 
        self.auto_chk = wx.CheckBox(panel, label="Use live CARLA speed + lead distance for ADAS")
        self.auto_chk.SetValue(self.carla_sim.vehicle is not None)
        self.auto_chk.Enable(self.carla_sim.vehicle is not None)
        self.head_control_chk = wx.CheckBox(panel, label="Drive with head movement (yaw steers, pitch pedals)")
        self.head_control_chk.SetValue(self.carla_sim.vehicle is not None)
        self.head_control_chk.Enable(self.carla_sim.vehicle is not None)
        self.autopilot_chk = wx.CheckBox(panel, label="CARLA autopilot")
        self.autopilot_chk.SetValue(self.carla_sim.autopilot)
        self.autopilot_chk.Enable(self.carla_sim.vehicle is not None)
        self.drive_help = wx.StaticText(panel, label="Head drive: turn left/right to steer; look up to accelerate, down to brake")
        recenter_btn = wx.Button(panel, label="Re-center head (look straight at screen first)")
 
        console_label = wx.StaticText(panel, label="Smart Command Console")
        self.command_input = wx.TextCtrl(panel, style=wx.TE_PROCESS_ENTER)
        self.command_input.SetHint("Ex: 'Set speed at 80 and distance 20'")
 
        self.speed_input = wx.TextCtrl(panel, value="0")
        self.distance_input = wx.TextCtrl(panel, value="30")
        self.obstacle_choice = wx.Choice(panel, choices=["Vehicle", "Animal", "Pothole"])
        self.obstacle_choice.SetSelection(0)
        self.road_choice = wx.Choice(panel, choices=["Dry", "Wet", "Poor"])
        self.road_choice.SetSelection(0)
 
        evaluate_btn = wx.Button(panel, label="Evaluate ADAS (log)")
        self.voice_btn = wx.Button(panel, label="Voice Command")
 
        self.ttc_output = wx.StaticText(panel, label="TTC: -- seconds")
        self.risk_output = wx.StaticText(panel, label="Risk Score: --")
        self.decision_output = wx.StaticText(panel, label="Decision: --")
        self.decision_output.SetFont(bold(12))
 
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(title, 0, wx.ALL | wx.CENTER, 8)
        sizer.Add(self.driver_label, 0, wx.ALL | wx.CENTER, 3)
        sizer.Add(self.head_label, 0, wx.ALL | wx.CENTER, 3)
        sizer.Add(recenter_btn, 0, wx.ALL | wx.CENTER, 5)
        sizer.Add(self.auto_chk, 0, wx.LEFT | wx.TOP, 20)
        sizer.Add(self.head_control_chk, 0, wx.LEFT | wx.TOP, 20)
        sizer.Add(self.autopilot_chk, 0, wx.LEFT | wx.TOP, 20)
        sizer.Add(self.drive_help, 0, wx.LEFT | wx.TOP, 20)
        sizer.Add(console_label, 0, wx.LEFT | wx.TOP, 20)
        sizer.Add(self.command_input, 0, wx.EXPAND | wx.ALL, 8)
        sizer.Add(wx.StaticLine(panel), 0, wx.EXPAND | wx.ALL, 5)
        for text, ctl in [("Vehicle Speed (km/h)", self.speed_input),
                          ("Distance to Obstacle (m)", self.distance_input),
                          ("Obstacle Type", self.obstacle_choice),
                          ("Road Condition", self.road_choice)]:
            sizer.Add(wx.StaticText(panel, label=text), 0, wx.LEFT, 20)
            sizer.Add(ctl, 0, wx.EXPAND | wx.ALL, 8)
        sizer.Add(evaluate_btn, 0, wx.ALL | wx.CENTER, 10)
        sizer.Add(self.ttc_output, 0, wx.ALL | wx.CENTER, 4)
        sizer.Add(self.risk_output, 0, wx.ALL | wx.CENTER, 4)
        sizer.Add(self.decision_output, 0, wx.ALL | wx.CENTER, 8)
        sizer.Add(self.voice_btn, 0, wx.ALL | wx.CENTER, 5)
        panel.SetSizer(sizer)
 
        self._apply_auto_state()
        self.auto_chk.Bind(wx.EVT_CHECKBOX, lambda e: self._apply_auto_state())
        self.head_control_chk.Bind(wx.EVT_CHECKBOX, self.on_head_control_toggle)
        self.autopilot_chk.Bind(wx.EVT_CHECKBOX, self.on_autopilot_toggle)
        recenter_btn.Bind(wx.EVT_BUTTON, lambda e: self.tracker.recalibrate())
        self.command_input.Bind(wx.EVT_TEXT_ENTER, self.on_command)
        evaluate_btn.Bind(wx.EVT_BUTTON, self.on_evaluate)
        self.voice_btn.Bind(wx.EVT_BUTTON, self.on_voice_input)
        self.Bind(wx.EVT_CLOSE, self.on_close)
 
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_tick, self.timer)
        self.timer.Start(50)       # 20 Hz control loop
        self.Show()
 
    # ---------- helpers ---------- #
    def _is_auto(self):
        return self.auto_chk.GetValue() and self.carla_sim.vehicle is not None
 
    def _apply_auto_state(self):
        manual = not self._is_auto()
        self.speed_input.Enable(manual)
        self.distance_input.Enable(manual)

    def on_autopilot_toggle(self, _event):
        enabled = self.autopilot_chk.GetValue()
        if enabled:
            self.head_control_chk.SetValue(False)
        self.carla_sim.set_autopilot(enabled)

    def on_head_control_toggle(self, _event):
        if self.head_control_chk.GetValue():
            self.autopilot_chk.SetValue(False)
            self.carla_sim.set_autopilot(False)

    @staticmethod
    def _head_pedal(pitch):
        if pitch >= HEAD_PEDAL_DEADZONE:
            return min(0.6, 0.6 * (pitch - HEAD_PEDAL_DEADZONE) /
                       (HEAD_PEDAL_FULL - HEAD_PEDAL_DEADZONE)), 0.0
        if pitch <= -HEAD_PEDAL_DEADZONE:
            return 0.0, min(0.8, 0.8 * (-pitch - HEAD_PEDAL_DEADZONE) /
                             (HEAD_PEDAL_FULL - HEAD_PEDAL_DEADZONE))
        return 0.0, 0.0

    @staticmethod
    def _head_steer(yaw):
        if abs(yaw) <= HEAD_STEER_DEADZONE:
            return 0.0
        magnitude = min(1.0, (abs(yaw) - HEAD_STEER_DEADZONE) /
                        (HEAD_STEER_FULL - HEAD_STEER_DEADZONE))
        return math.copysign(magnitude, yaw)

    @staticmethod
    def _key_down(vk_code):
        try:
            return bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000)
        except (AttributeError, OSError):
            return False

    def _manual_keys(self):
        forward = self._key_down(ord("W")) or self._key_down(0x26)  # W / Up
        back = self._key_down(ord("S")) or self._key_down(0x28)     # S / Down
        left = self._key_down(ord("A")) or self._key_down(0x25)     # A / Left
        right = self._key_down(ord("D")) or self._key_down(0x27)    # D / Right
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
            self.distance_input.ChangeValue(f"{di:.1f}")
            return sp, di
        try:
            sp, di = float(self.speed_input.GetValue()), float(self.distance_input.GetValue())
        except ValueError:
            return None
        return (sp, di) if di > 0 else None
 
    # ---------- main control loop ---------- #
    def on_tick(self, _event):
        if self.tracker.error:
            self._set(self.driver_label, f"Driver State: unavailable ({self.tracker.error})")
            if not self._camera_warned:
                self._camera_warned = True
                wx.MessageBox(f"{self.tracker.error}. Driver monitoring is off.", "Camera")
            snap = dict(state="normal", face=False, yaw=0.0, pitch=0.0, gaze_yaw=0.0, calibrated=False)
        else:
            snap = self.tracker.snapshot()
        state = snap["state"]
 
        # 1. Head + eye movement -> interior camera in CARLA (always at the driver's eyes)
        if snap["face"] and snap["calibrated"]:
            ty = max(-MAX_VIEW_YAW, min(MAX_VIEW_YAW,
                     snap["yaw"] * YAW_GAIN + snap["gaze_yaw"] * GAZE_VIEW_WEIGHT))
            tp = max(-MAX_VIEW_PITCH, min(MAX_VIEW_PITCH, snap["pitch"] * PITCH_GAIN))
            self.view_yaw += VIEW_SMOOTHING * (ty - self.view_yaw)
            self.view_pitch += VIEW_SMOOTHING * (tp - self.view_pitch)
        # (face lost -> hold last view instead of snapping)
        self.carla_sim.update_view(self.view_yaw, self.view_pitch)
 
        # 2. ADAS evaluation
        decision = "MONITOR"
        inputs = self._read_inputs()
        if inputs:
            sp, di = inputs
            obstacle = self.obstacle_choice.GetStringSelection()
            road = self.road_choice.GetStringSelection()
            ttc, risk, decision = evaluate_adas(sp, di, obstacle, road, state)
            self._set(self.ttc_output, f"TTC: {ttc} seconds")
            self._set(self.risk_output, f"Risk Score: {risk}")
            self._set(self.decision_output, f"Decision: {decision}")
            key = (decision, state)
            if key != self._last_key:
                self._last_key = key
                log_scenario(sp, di, obstacle, road, state, ttc, risk, decision)
 
        # 3. Actuate in the simulator + alert the driver
        adas_brake = brake_level(decision, state, self._is_auto())
        if self.autopilot_chk.GetValue():
            self.carla_sim.set_brake(adas_brake)
        elif self.head_control_chk.GetValue():
            if snap["face"] and snap["calibrated"]:
                throttle, brake = self._head_pedal(snap["pitch"])
                steer = self._head_steer(snap["yaw"])
            else:
                throttle, steer, brake = 0.0, 0.0, 0.8
            self.carla_sim.apply_manual_control(
                throttle=throttle,
                steer=steer,
                brake=max(brake, adas_brake))
        else:
            throttle, steer, brake = self._manual_keys()
            self.carla_sim.apply_manual_control(
                throttle=throttle,
                steer=steer,
                brake=max(brake, adas_brake))
        now = time.time()
        if state == "drowsy" and now - self._last_beep > 1.0:
            beep_async(1000, 300)
            self._last_beep = now
        elif state == "distracted" and now - self._last_beep > 2.0:
            beep_async(600, 200)
            self._last_beep = now
 
        # 4. Status labels + preview windows
        self._set(self.driver_label, f"Driver State: {state.upper()}")
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
 
    # ---------- buttons / commands ---------- #
    def on_evaluate(self, _event):
        inputs = self._read_inputs()
        if not inputs:
            wx.MessageBox("Please enter valid numbers (distance > 0)", "Input Error")
            return
        sp, di = inputs
        state = self.tracker.snapshot()["state"]
        obstacle = self.obstacle_choice.GetStringSelection()
        road = self.road_choice.GetStringSelection()
        ttc, risk, decision = evaluate_adas(sp, di, obstacle, road, state)
        self._set(self.ttc_output, f"TTC: {ttc} seconds")
        self._set(self.risk_output, f"Risk Score: {risk}")
        self._set(self.decision_output, f"Decision: {decision}")
        log_scenario(sp, di, obstacle, road, state, ttc, risk, decision)
 
    def on_command(self, _event):
        parsed = parse_command(self.command_input.GetValue().lower())
        if "speed" in parsed or "distance" in parsed:
            self.auto_chk.SetValue(False)          # typed numbers mean "what-if" mode
            self._apply_auto_state()
        if "speed" in parsed:
            self.speed_input.SetValue(parsed["speed"])
        if "distance" in parsed:
            self.distance_input.SetValue(parsed["distance"])
        if "obstacle" in parsed:
            self.obstacle_choice.SetStringSelection(parsed["obstacle"])
        if "road" in parsed:
            self.road_choice.SetStringSelection(parsed["road"])
        self.on_evaluate(None)
 
    def on_voice_input(self, _event):
        self.voice_btn.Disable()
        self.voice_btn.SetLabel("Listening...")
        threading.Thread(target=lambda: wx.CallAfter(self._voice_done, listen_once()), daemon=True).start()
 
    def _voice_done(self, text):
        self.voice_btn.Enable()
        self.voice_btn.SetLabel("Voice Command")
        if text:
            self.command_input.SetValue(text)
            self.on_command(None)
        else:
            wx.MessageBox("Didn't catch that (check mic / internet).", "Voice")
 
    def on_close(self, _event):
        self.timer.Stop()
        self.tracker.stop()
        if self.vision:
            self.vision.stop()
        self.carla_sim.cleanup()
        cv2.destroyAllWindows()
        self.Destroy()
 
 
# =====================================================================
# 8. BOOTSTRAP
# =====================================================================
if __name__ == "__main__":
    app = wx.App()
    ADASApp()
    app.MainLoop()
 
