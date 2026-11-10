"""Shared, public experiment settings. Hidden scene contents live in scene/build.py."""
from pathlib import Path
from dataclasses import dataclass, asdict
import json

ROOT = Path(__file__).resolve().parent

@dataclass(frozen=True)
class Config:
    measurement_mode: str = 'reference'
    online_decisions: bool = True
    lidar_period: float = .05
    motor_calibration_error: float = 0.
    imu_acc_noise: float = .015
    imu_gyro_noise: float = .00015
    cable_length_noise: float = .0001
    lidar_range_noise: float = .002
    motor_thrust_noise: float = .002
    probe_action: str = "guarded"
    live_belief_period: float = .15
    timestep: float = .002
    flight_speed: float = .65
    contact_approach_speed: float = .30
    contact_slow_band_m: float = .34
    contact_braking_accel: float = .45
    local_merge_distance_m: float = 2.00
    cruise_lookahead_m: float = .55
    preview_period_s: float = .35
    preview_position_change_m: float = .04
    preview_object_change_m: float = .01
    clearance: float = .34
    grid_resolution: float = .05
    sobol_power: int = 10
    inertia_min: float = .05
    inertia_max: float = .80
    friction_min: float = .01
    friction_max: float = 4.  # N m; candidate upper bound (2026-10-28 adjustment)
    hinge_damping: float = .05
    probe_duration: float = 1.2
    probe_force_max: float = 2.2
    push_force_max: float = 2.2
    push_timeout: float = 10.
    force_scale: float = .30
    angle_scale: float = .035
    alpha: float = 2.
    geometry_threshold: float = 3.
    disagreement_threshold: float = .36
    sensor_period: float = .1
    camera_width: int = 480
    camera_height: int = 320
    camera_fovy: float = 68.
    scan_range: float = 6.
    scan_rays: int = 161
    confirmation_frames: int = 3
    yolo_model: str = 'models/vision/door_glass_box_yolov8n.onnx'
    yolo_input_size: int = 320
    yolo_confidence: float = .45
    yolo_iou: float = .45
    vision_max_age: float = .65
    vision_max_gap: float = .35
    vision_wait_s: float = .07
    scan_history_s: float = 2.
    ambiguity_min_side_points: int = 3
    ambiguity_gap_fraction: float = .45
    glass_surface_return_probability: float = .12
    glass_dropout_probability: float = .18
    verify_speed: float = .10
    verify_timeout: float = 10.
    verify_force_floor: float = .45
    verify_force_sigma: float = 5.
    force_filter_time: float = .025
    workers: int = 4
    seed: int = 22

    # Dynamics method only. Angles in rad, torques N m, inertia kg m^2.
    THETA_CLEAR: float = 1.3089969389957472  # legacy/math-only fallback; live dynamics uses computed geometry
    B_REF: float = 5.235987755982989  # fixed J reference, independent of computed opening angle
    DOOR_POSITION_MARGIN: float = .025  # m; tracking/localization allowance beyond collision geometry
    DOOR_FLEX_MARGIN: float = .025  # m; additional multibody deformation allowance
    DOOR_PASS_TILT_DEG: float = 15.
    DOOR_PASS_YAW_DEG: float = 7.
    DOOR_ANGLE_MARGIN_DEG: float = 2.  # observed panel-angle allowance
    DOOR_CLEARANCE_STEP_DEG: float = 1.  # geometric search resolution, not an opening target
    DOOR_PASS_SPEED: float = .35
    DOOR_PASS_MAX_OMEGA: float = .15
    DOOR_PASS_LOOKAHEAD: float = .25  # s; swept-angle allowance during passage
    DOOR_CLEAR_HEIGHT: float = 2.04  # public lintel lower surface in scene/build.py
    TAU_F_REF: float = 4.  # normalization reference, NOT a true-door parameter
    DECISION_LAMBDA: float = 1.  # dimensionless path-saving/effort tradeoff
    TAU_PUSH_MAX: float = None  # None -> R_PUSH * push_force_max
    R_PUSH: float = .835  # m; nominal controller contact arm (.24375 + 1.1825/2)
    SIGMA_THETA: float = .035  # rad; trajectory mismatch scale
    SIGMA_OMEGA: float = .15  # rad/s; optional velocity mismatch scale
    OMEGA_ERROR_WEIGHT: float = 0.  # first version uses angle only
    OMEGA_STICK_EPS: float = 1e-5  # rad/s; static-friction velocity tolerance
    DOOR_INTEGRATION_MAX_STEP: float = .002  # s; substep bound
    DOOR_ANGLE_MIN: float = 0.  # public hinge stop, same as unchanged scene
    DOOR_ANGLE_MAX: float = 1.53588974175501  # 88 deg, same as unchanged scene
    PUSH_CONFIDENCE: float = .8
    AVOID_CONFIDENCE: float = .2
    SIGMA_DELTAJ_PROBE_THRESHOLD: float = .15  # require material utility spread
    DELTAJ_NEAR_ZERO_THRESHOLD: float = .03  # conservative utility tie tolerance
    D_PROBE_THRESHOLD: float = .5  # ALSO require sigma; D alone never probes

    @property
    def b_ref(self):
        return 5.235987755982989 if self.B_REF is None else self.B_REF

    @property
    def tau_push_max(self):
        return self.R_PUSH * self.push_force_max if self.TAU_PUSH_MAX is None else self.TAU_PUSH_MAX

def architecture():
    return json.loads((ROOT/'config'/'architecture.json').read_text(encoding='utf-8'))

def load_config():
    p = ROOT/'config'/'settings.json'
    return Config(**json.loads(p.read_text(encoding='utf-8'))) if p.exists() else Config()
