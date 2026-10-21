# resumeprojectpart2
this is my second real adas project not the real one but a version which i tried in 4 yrs of engineering as i learnt skills

## Shorter standalone version

`ad_compact.py` keeps driver monitoring/calibration, head steering, keyboard
controls, CARLA autopilot, YOLO road previews, typed/voice commands, and CSV logs.
It shares form construction, evaluation, and worker code instead of repeating them.
The original `ad.py` is preserved.

Run in the Python environment used for `ad.py`:

```bash
python ad_compact.py
```

Dependencies are the same: PyTorch, Ultralytics, OpenCV, NumPy,
MediaPipe 0.10.14, wxPython, and SpeechRecognition (with microphone support).
CARLA is optional; configure `CARLA_HOST` and `CARLA_PORT` for a running server.
Without CARLA, manual scenario evaluation and driver monitoring are available.
Voice recognition uses Google's service and needs internet access.
YOLO annotations are visual only; live lead distances come from CARLA vehicles.
The risk score is a demo heuristic, not a validated collision model.

The compact version stops head acceleration on failed/stalled camera frames,
represents a clear road without inventing an obstacle, and connects directly to
local or remote CARLA servers. Missing driver monitoring commands a controlled
stop in the simulator. Re-center while looking straight before head driving.

Logic checks use synthetic dependencies and need only Python:

```bash
python -m unittest discover -s tests -v
```

These checks do not replace testing with a real webcam, GUI, and CARLA server.
