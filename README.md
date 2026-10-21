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

## Chase camera, gentler driving, and clearer logs

The CARLA spectator now follows 6 m behind the vehicle center and 3.2 m above it,
looking down 15 degrees over the trunk. Head movement no longer rotates the view.
The separate YOLO road preview stays on the forward sensor.

Head driving starts **off**. Look straight, press **Re-center head**, wait for
calibration, then enable **Drive with head movement**. Small turns below 8 degrees
are ignored; maximum head steering is 45%, reached at 40 degrees. Steering and
throttle increase gradually, while braking and tracking-loss stops act immediately.
Tune the `HEAD_*` constants near the top of `ad_compact.py` if needed.
Typing into text fields suppresses keyboard driving.

New runs write `adas_log_v3.csv`, preserving older `adas_log.csv` and
`adas_log_v2.csv` files. Rows are
recorded at least once per second while inputs are valid, plus state/action changes.
`risk_decision` describes the risk heuristic; `control_action` describes the control
command (for example, `DROWSY STOP` with `commanded_brake=0.8`). These values are
commands, not measured proof that the car braked. Autopilot-managed controls and
`EVALUATION_ONLY` rows leave command values blank. `NO_LEAD_IN_RANGE` means no
vehicle was found within the 80 m search corridor; TTC is `N/A` when stopped or
no lead is found. This does not prove the entire road is clear.

## Drive only while looking straight

1. Sit normally with your eyes open and look straight at the screen.
2. Press **Re-center head** and wait until calibration completes.
3. Enable **Drive only while looking straight (autopilot steers)**.

CARLA autopilot handles the road and turns. The car is allowed to move after
looking forward continuously for 0.75 seconds; returning from a stop also respects
the existing 1.5-second brake hold. Looking away, tilting too far, losing the face,
recalibrating, or a drowsiness warning commands braking. Looking up does not
accelerate and head yaw does not steer in this mode. The other head/keyboard/
autopilot modes remain selectable separately.

Forward entry tolerances are 10 degrees yaw, 12 degrees pitch, and 12 degrees
horizontal eye gaze. Slightly wider exit tolerances prevent flicker at the
boundary. Tune `STRAIGHT_ENTER`, `STRAIGHT_EXIT`, and `STRAIGHT_HOLD_S` if needed.
This uses head pose and horizontal gaze; vertical eye gaze alone is not measured.

The eye-closure threshold is now calibrated from your median open-eye EAR, at 60%
of that baseline (capped at 0.21). A sustained closure still needs 2 seconds to
trigger drowsiness. Extreme head angles are treated as distraction rather than
using an unreliable eye measurement. Calibration must be done with eyes open;
these heuristics still need checking against your camera and face.

`adas_log_v3.csv` also records yaw, pitch, horizontal gaze, EAR, and the calibrated
EAR threshold. These fields help investigate false drowsiness reports. Earlier
CSV files keep their original format.
