"""Regression checks using synthetic camera/CARLA dependencies, no hardware needed."""
import ast
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def load_demo():
    fake_modules = {name: Mock() for name in (
        'torch', 'ultralytics', 'cv2', 'mediapipe', 'speech_recognition', 'carla')}
    fake_modules['mediapipe'] = MagicMock()
    fake_modules['numpy'] = NS(array=lambda value, **kw: value, float64=float)
    fake_modules['wx'] = NS(Frame=type('Frame', (), {
        '__init__': lambda self, *a, **kw: None,
        'Bind': lambda *a: None, 'Show': lambda *a: None, 'Destroy': lambda *a: None}))
    spec = importlib.util.spec_from_file_location('ad_compact_test', ROOT / 'ad_compact.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, fake_modules):
        spec.loader.exec_module(module)
    return module


class CompactTests(unittest.TestCase):
    def setUp(self):
        self.app = load_demo()

    def simulator(self, positions=()):
        vec = lambda x, y: NS(x=x, y=y)
        tf = NS(get_forward_vector=lambda: vec(1, 0),
                get_right_vector=lambda: vec(0, 1), location=vec(0, 0))
        ego = NS(id=1, get_transform=lambda: tf)
        actors = [ego] + [NS(id=i+2, get_location=lambda p=p: vec(*p))
                          for i, p in enumerate(positions)]
        sim = self.app.CarlaIntegration.__new__(self.app.CarlaIntegration)
        sim.vehicle = ego
        sim.world = NS(get_actors=lambda: NS(filter=lambda _: actors))
        sim.npcs = []
        sim._lead_cache = (0.0, math.inf)
        return sim

    def test_clear_road_does_not_brake(self):
        distance = self.simulator().get_lead_distance()
        self.assertEqual(distance, math.inf)
        for speed in (0, 60, 120, 200):
            _, risk, decision = self.app.evaluate_adas(speed, distance, 'Vehicle', 'Dry', 'normal')
            self.assertEqual((risk, decision), (0.0, 'MONITOR'))
            self.assertEqual(self.app.brake_level(decision, 'normal', True), 0)

    def test_lead_distance_filters_ego_behind_and_adjacent_vehicles(self):
        self.assertEqual(self.simulator([(-10, 0), (10, 3), (20, 0)]).get_lead_distance(), 15.5)
        self.assertEqual(self.simulator([(2, 0)]).get_lead_distance(), 0.5)

    def test_original_risk_decisions_preserved_for_real_obstacles(self):
        original = ast.parse((ROOT / 'ad.py').read_text())
        func = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate_adas')
        ns = {}
        exec(compile(ast.Module(body=[func], type_ignores=[]), 'original', 'exec'), ns)
        for speed in (10, 60, 120):
            for distance in (0.5, 5, 30, 80):
                for road in ('Dry', 'Wet', 'Poor'):
                    for obstacle in ('Vehicle', 'Animal', 'Pothole'):
                        for state in ('normal', 'drowsy', 'distracted'):
                            args = speed, distance, obstacle, road, state
                            self.assertEqual(self.app.evaluate_adas(*args), ns['evaluate_adas'](*args))

    def test_invalid_numbers_rejected(self):
        for speed, distance in ((-1, 30), (math.nan, 30), (math.inf, 30), (10, 0), (10, -1), (10, math.nan)):
            with self.assertRaises(ValueError):
                self.app.evaluate_adas(speed, distance, 'Vehicle', 'Dry', 'normal')

    def test_head_controls_and_tracking_loss(self):
        snap = dict(face=True, calibrated=True, yaw=25, pitch=25)
        self.assertEqual(self.app.head_controls(snap), (0.6, 1.0, 0))
        self.assertEqual(self.app.head_controls(dict(snap, yaw=-25, pitch=-25)), (0, -1, 0.8))
        self.assertEqual(self.app.head_controls(dict(snap, face=False)), (0, 0, 0.8))
        self.assertEqual(self.app.head_controls(dict(snap, calibrated=False)), (0, 0, 0.8))
        self.assertEqual(self.app.brake_level('MONITOR', 'unavailable', True), 0.8)
        self.assertEqual(self.app.brake_level('EMERGENCY BRAKE', 'normal', False), 0)

    def test_stalled_frame_invalidated(self):
        tracker = self.app.DriverTracker()
        tracker._snap.update(face=True, calibrated=True, pitch=25)
        tracker._updated_at = self.app.time.monotonic() - 1
        self.assertEqual(self.app.head_controls(tracker.snapshot()), (0, 0, 0.8))

    def test_failed_reads_invalidate_controls_and_release_camera(self):
        tracker = self.app.DriverTracker()
        tracker._snap.update(face=True, calibrated=True, pitch=25)
        tracker._updated_at = self.app.time.monotonic()
        def failed_read():
            tracker._stop_evt.set()
            return False, None
        cap = Mock()
        cap.read.side_effect = failed_read
        self.app.cv2.VideoCapture.return_value = cap
        tracker.run()
        cap.read.assert_called_once()
        self.assertIsNone(tracker.error)
        self.assertEqual(self.app.head_controls(tracker.snapshot()), (0, 0, 0.8))
        cap.release.assert_called_once()

    def test_recalibration_invalidates_previous_pose(self):
        tracker = self.app.DriverTracker()
        tracker._snap.update(face=True, calibrated=True, pitch=25)
        tracker._updated_at = self.app.time.monotonic()
        tracker.recalibrate()
        self.assertEqual(self.app.head_controls(tracker.snapshot()), (0, 0, 0.8))

    def test_remote_carla_contacted_directly(self):
        client, world = Mock(), Mock()
        client.get_world.return_value = world
        self.app.carla.Client.return_value = client
        self.assertEqual(self.app.connect_carla('192.0.2.1', 2000), (client, world))
        self.app.carla.Client.assert_called_once_with('192.0.2.1', 2000)

    def test_carla_retry_timeout(self):
        client = Mock()
        client.get_world.side_effect = RuntimeError('not ready')
        self.app.carla.Client.return_value = client
        clock = NS(now=0.0)
        fake_time = NS(monotonic=lambda: clock.now,
                       sleep=lambda seconds: setattr(clock, 'now', clock.now + seconds))
        with patch.object(self.app, 'time', fake_time):
            with self.assertRaisesRegex(RuntimeError, '192.0.2.1:2000'):
                self.app.connect_carla('192.0.2.1', 2000, timeout=1)
        self.assertEqual(clock.now, 1)

    def test_commands(self):
        self.assertEqual(self.app.parse_command('Set speed at 80 and distance 20 on wet road'),
                         dict(speed='80', distance='20', road='Wet'))
        self.assertEqual(self.app.parse_command('animal on dry road'), dict(obstacle='Animal', road='Dry'))

    def test_manual_input_validation(self):
        gui = self.app.ADASApp.__new__(self.app.ADASApp)
        gui._is_auto = lambda: False
        gui.speed_input, gui.distance_input = Mock(), Mock()
        for speed, distance in [('nan', '30'), ('80', 'inf'), ('-2', '30'), ('80', '0'), ('x', '30')]:
            gui.speed_input.GetValue.return_value = speed
            gui.distance_input.GetValue.return_value = distance
            self.assertIsNone(gui._read_inputs())
        gui.speed_input.GetValue.return_value = '80'
        gui.distance_input.GetValue.return_value = '30'
        self.assertEqual(gui._read_inputs(), (80, 30))

    def test_live_clear_distance_display(self):
        gui = self.app.ADASApp.__new__(self.app.ADASApp)
        gui._is_auto = lambda: True
        gui.carla_sim = Mock()
        gui.carla_sim.get_speed.return_value = 120
        gui.carla_sim.get_lead_distance.return_value = math.inf
        gui.speed_input, gui.distance_input = Mock(), Mock()
        self.assertEqual(gui._read_inputs(), (120, math.inf))
        gui.distance_input.ChangeValue.assert_called_once_with('Clear')

    def test_shared_evaluation_updates_display_and_logs_transitions(self):
        gui = self.app.ADASApp.__new__(self.app.ADASApp)
        gui._read_inputs = lambda: (80, 20)
        gui.obstacle_choice, gui.road_choice = Mock(), Mock()
        gui.obstacle_choice.GetStringSelection.return_value = 'Vehicle'
        gui.road_choice.GetStringSelection.return_value = 'Dry'
        gui.ttc_output, gui.risk_output, gui.decision_output = Mock(), Mock(), Mock()
        gui._last_key = None
        with patch.object(self.app, 'log_scenario') as log:
            expected = self.app.evaluate_adas(80, 20, 'Vehicle', 'Dry', 'normal')[2]
            self.assertEqual(gui.evaluate('normal'), expected)
            gui.evaluate('normal')
            self.assertEqual(log.call_count, 1)
            gui.evaluate('normal', force_log=True)
            self.assertEqual(log.call_count, 2)
        gui.decision_output.SetLabel.assert_called_with('Decision: ' + expected)

    def test_form_command_timer_and_shutdown_wiring(self):
        def widget(parent=None, value='', label='', choices=None, **kw):
            control = Mock()
            state = dict(value=value, label=label, selection=0)
            control.GetValue.side_effect = lambda: state['value']
            control.SetValue.side_effect = lambda v: state.update(value=v)
            control.ChangeValue.side_effect = control.SetValue.side_effect
            control.GetLabel.side_effect = lambda: state['label']
            control.SetLabel.side_effect = lambda v: state.update(label=v)
            control.SetSelection.side_effect = lambda v: state.update(selection=v)
            control.GetStringSelection.side_effect = lambda: choices[state['selection']]
            return control
        fake_wx = Mock()
        for flag in ('VERTICAL', 'EXPAND', 'ALL', 'TE_PROCESS_ENTER'):
            setattr(fake_wx, flag, 0)
        for name in ('StaticText', 'TextCtrl', 'Choice', 'CheckBox', 'Button'):
            getattr(fake_wx, name).side_effect = widget
        tracker, sim = Mock(), Mock()
        sim.vehicle = None
        tracker.error = None
        tracker.snapshot.return_value = dict(state='normal', face=False, calibrated=False)
        tracker.get_frame.return_value = None
        with patch.object(self.app, 'wx', fake_wx), patch.object(self.app, 'DriverTracker', return_value=tracker), \
                patch.object(self.app, 'CarlaIntegration', return_value=sim), \
                patch.object(self.app, 'log_scenario'):
            gui = self.app.ADASApp()
            tracker.start.assert_called_once()
            gui.command_input.SetValue('speed 80 distance 20')
            gui.on_command(None)
            self.assertEqual(gui._read_inputs(), (80, 20))
            gui._manual_keys = lambda: (0, 0, 0)
            gui.on_tick(None)
            self.assertEqual(gui.driver_label.GetLabel(), 'Driver State: NORMAL')
            sim.apply_manual_control.assert_called_once_with(0, 0, 0)
            gui.on_close(None)
            tracker.stop.assert_called_once()
            sim.cleanup.assert_called_once()

    def test_manual_brake_suppresses_throttle(self):
        sim = self.simulator()
        sim.autopilot = False
        sim.vehicle.apply_control = Mock()
        self.app.carla.VehicleControl.side_effect = lambda **kw: kw
        sim.apply_manual_control(throttle=0.6, steer=2, brake=0.8)
        sim.vehicle.apply_control.assert_called_once_with(dict(throttle=0, steer=1, brake=0.8))

    def test_autopilot_resumes_after_brake_hold(self):
        sim = self.simulator()
        sim.autopilot = True
        sim.vehicle.apply_control, sim.vehicle.set_autopilot = Mock(), Mock()
        with patch.object(self.app.time, 'time', return_value=100):
            sim.set_brake(0.8)
            sim.set_brake(0)
            self.assertFalse(sim.autopilot)
        with patch.object(self.app.time, 'time', return_value=102):
            sim.set_brake(0)
            self.assertTrue(sim.autopilot)


if __name__ == '__main__':
    unittest.main()
