"""
Unified ADAS Eye-Gaze Vehicle Control & Drowsiness Dashboard (CARLA Hand-Free Edition)
---------------------------------------------------------------------------------
Monitors your hardware webcam to handle hands-free driving and safety:
  1. Driver Monitoring: Tracks EAR. If eyes stay closed for > 2 seconds, 
     it flags a 'DROWSY' state, sounds an audible alarm, and overrides control to brake.
  2. Gaze Control: Maps face tilt orientation directly to car mechanics.
  3. Automatic Camera Stream: Spawns a physical camera sensor welded to the vehicle 
     so the simulation view automatically moves with the car.
"""

import os
import random
import time
import cv2
import mediapipe as mp
import numpy as np
import wx

# Cross-platform system alarm speaker support
try:
    import winsound
except ImportError:
    winsound = None

try:
    import carla
except ImportError:
    carla = None


# ============================================================= #
# CONFIGURATION
# ============================================================= #
CARLA_HOST = "localhost"
CARLA_PORT = 2000

EAR_THRESHOLD = 0.21
EYES_CLOSED_DROWSY_SECONDS = 2.0

LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]


# ============================================================= #
# ADAS DRIVER MONITORING & GAZE ENGINE
# ============================================================= #
class IntegratedDriverController:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.7)
        
        # State tracking variables
        self.closed_start_time = None
        self.driver_state = "normal"  
        self.current_command = "STOP"

    @staticmethod
    def _calculate_ear(landmarks, idx):
        pt = lambda i: np.array([landmarks[idx[i]].x, landmarks[idx[i]].y])
        v1 = np.linalg.norm(pt(1) - pt(5))
        v2 = np.linalg.norm(pt(2) - pt(4))
        h = np.linalg.norm(pt(0) - pt(3))
        return (v1 + v2) / (2.0 * h) if h else 0.0

    def trigger_alarm(self):
        """Sounds a high-frequency alert to awaken the driver."""
        if winsound:
            winsound.Beep(2000, 250)
        else:
            print('\a', end='', flush=True)

    def evaluate_frame(self, frame):
        if frame is None:
            return "STOP", "normal"

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return "STOP", self.driver_state

        landmarks = results.multi_face_landmarks[0].landmark

        # --- PIPELINE 1: Drowsiness Evaluation (EAR) ---
        ear = (self._calculate_ear(landmarks, LEFT_EYE_IDX) + self._calculate_ear(landmarks, RIGHT_EYE_IDX)) / 2.0
        eyes_open = ear >= EAR_THRESHOLD

        if not eyes_open:
            if self.closed_start_time is None:
                self.closed_start_time = time.time()
            elif time.time() - self.closed_start_time > EYES_CLOSED_DROWSY_SECONDS:
                self.driver_state = "drowsy"
                self.trigger_alarm()  
        else:
            self.closed_start_time = None
            self.driver_state = "normal"

        # --- SAFETY OVERRIDE ---
        if self.driver_state == "drowsy":
            self.current_command = "EMERGENCY STOP (DROWSY)"
            return self.current_command, self.driver_state

        # --- PIPELINE 2: Gaze Direction Mapping (When Awake) ---
        nose = np.array([landmarks[1].x * w, landmarks[1].y * h])
        l_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
        r_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
        chin = np.array([landmarks[152].x * w, landmarks[152].y * h])
        forehead = np.array([landmarks[10].x * w, landmarks[10].y * h])

        # Direction shifts ratios
        horizontal_ratio = np.linalg.norm(nose - l_eye) / (np.linalg.norm(nose - r_eye) or 1)
        vertical_ratio = np.linalg.norm(nose - forehead) / (np.linalg.norm(nose - chin) or 1)

        if horizontal_ratio > 1.35:
            self.current_command = "TURN LEFT"
        elif horizontal_ratio < 0.75:
            self.current_command = "TURN RIGHT"
        elif vertical_ratio > 1.45:  # Calibrated wider to fit 45-degree cooling pads
            self.current_command = "MOVE FRONT"
        elif vertical_ratio < 0.70:  # Calibrated wider to fit 45-degree cooling pads
            self.current_command = "MOVE BACK"
        else:
            self.current_command = "STOP"

        return self.current_command, self.driver_state


# ============================================================= #
# CARLA SIMULATOR INTERFACE WITH AUTO-CAMERA STREAM
# ============================================================= #
class CarlaManager:
    def __init__(self):
        self.connected = False
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera_sensor = None
        self.carla_frame = None

    def _process_camera_image(self, image):
        """Asynchronously unpacks streaming CARLA image buffers into OpenCV."""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        self.carla_frame = array[:, :, :3]  # Drop transparency alpha layer

    def connect(self):
        if carla is None:
            print("`carla` library missing. Running in standalone context.")
            return False

        try:
            self.client = carla.Client(CARLA_HOST, CARLA_PORT)
            self.client.set_timeout(4.0)
            self.world = self.client.get_world()
            
            # Spawn a controllable ego vehicle
            bp_lib = self.world.get_blueprint_library()
            vehicle_bp = bp_lib.filter("model3")[0]
            spawn_point = random.choice(self.world.get_map().get_spawn_points())
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            self.vehicle.set_autopilot(False)

            # --- SPAWN CHASE SENSOR CAMERA ATTACHED TO TRUNK ---
            camera_bp = bp_lib.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '800')
            camera_bp.set_attribute('image_size_y', '600')
            camera_bp.set_attribute('fov', '90')
            
            # Welded 5.5 meters behind and 2.8 meters above the roof line, looking slightly downwards
            camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
            self.camera_sensor = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
            
            # Route data stream pipeline hook
            self.camera_sensor.listen(lambda image: self._process_camera_image(image))
            
            self.connected = True
            print("Successfully hooked into CARLA simulation environment!")
            return True
        except Exception as e:
            print(f"Connection skipped: {e}. Falling back to standalone vision loop.")
            return False

    def send_drive_command(self, command):
        if not self.connected or not self.vehicle:
            return

        control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0, reverse=False)

        if command == "MOVE FRONT":
            control.throttle = 0.5
        elif command == "MOVE BACK":
            control.throttle = 0.3
            control.reverse = True
        elif command == "TURN LEFT":
            control.throttle = 0.25
            control.steer = -0.5
        elif command == "TURN RIGHT":
            control.throttle = 0.25
            control.steer = 0.5
        elif "STOP" in command:  
            control.brake = 1.0

        try:
            self.vehicle.apply_control(control)
        except Exception:
            self.connected = False

    def cleanup(self):
        if self.camera_sensor:
            self.camera_sensor.stop()
            self.camera_sensor.destroy()
        if self.vehicle:
            self.vehicle.destroy()


# ============================================================= #
# WXPYTHON DASHBOARD
# ============================================================= #
class ADASGazeApp(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Unified ADAS Dashboard Control Room", size=(500, 350))
        self.processor = IntegratedDriverController()
        self.carla_mgr = CarlaManager()
        self.cap = cv2.VideoCapture(0)

        panel = wx.Panel(self)
        panel.SetBackgroundColour("#121212")
        
        self.cmd_text = wx.StaticText(panel, label="Action: WAITING")
        self.cmd_text.SetForegroundColour("#00FF00")
        self.cmd_text.SetFont(wx.Font(16, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))

        self.state_text = wx.StaticText(panel, label="Driver State: INITIALIZING")
        self.state_text.SetForegroundColour("#FFFFFF")
        self.state_text.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        
        self.status_conn = wx.StaticText(panel, label="CARLA Engine Status: Connecting...")
        self.status_conn.SetForegroundColour("#888888")

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.cmd_text, 0, wx.ALL | wx.CENTER, 30)
        sizer.Add(self.state_text, 0, wx.ALL | wx.CENTER, 15)
        sizer.Add(self.status_conn, 0, wx.ALL | wx.CENTER, 25)
        panel.SetSizer(sizer)

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_tick, self.timer)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.Show()

        # Let GUI mount first to bypass terminal crash loop
        wx.CallAfter(self.init_carla)

    def init_carla(self):
        if self.carla_mgr.connect():
            self.status_conn.SetLabel("CARLA Engine Status: Online")
        else:
            self.status_conn.SetLabel("CARLA Engine Status: Offline Preview")
        self.timer.Start(50)  

    def on_tick(self, event):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            command, driver_state = self.processor.evaluate_frame(frame)
            
            self.cmd_text.SetLabel(f"Action: {command}")
            self.state_text.SetLabel(f"Driver Condition: {driver_state.upper()}")
            
            if driver_state == "drowsy":
                self.cmd_text.SetForegroundColour("#FF0000")
                self.state_text.SetForegroundColour("#FF0000")
            else:
                self.cmd_text.SetForegroundColour("#00FF00")
                self.state_text.SetForegroundColour("#FFFFFF")
            
            self.carla_mgr.send_drive_command(command)
            
            # Window 1: Hardware Webcam Monitoring 
            color = (0, 0, 255) if driver_state == "drowsy" else (0, 255, 0)
            cv2.putText(frame, f"CMD: {command}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"STATE: {driver_state.upper()}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("ADAS Driver Telemetry Feed", frame)

            # Window 2: Dynamic Live Auto-Follow Game Window
            if self.carla_mgr.carla_frame is not None:
                cv2.imshow("CARLA Hands-Free Driving View", self.carla_mgr.carla_frame)

    def on_close(self, event):
        if self.timer.IsRunning():
            self.timer.Stop()
        self.cap.release()
        self.carla_mgr.cleanup()
        cv2.destroyAllWindows()
        self.Destroy()


if __name__ == "__main__":
    app = wx.App()
    ADASGazeApp()
    app.MainLoop()