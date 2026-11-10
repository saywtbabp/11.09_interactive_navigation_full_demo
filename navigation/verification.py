"""Thesis-inspired low-speed verification, with explicit incomplete outcome.

Uses own state, calibrated IMU residual force and anonymous scans. Rotor control
is retained; this is not MINCO or a reproduction of the full thesis planner.
"""
import math
import numpy as np
from .physics import PhysicsFailure


def verify_opening(mission, rec, *, free_exit=None):
    w, sensors, cfg = mission.world, mission.sensors, mission.cfg
    evidence = rec.ambiguity_evidence or rec.range_evidence
    center = np.array(evidence.get('center_xy_m', rec.geometry['center_xy_m']), dtype=float)
    normal = np.array(evidence.get('normal_xy', [0., 1.]), dtype=float)
    normal /= max(1e-9, np.linalg.norm(normal))
    if (center-w.observed()[0][:2])@normal < 0: normal = -normal
    tangent = np.array([normal[1], -normal[0]])
    contact = center+.12*tangent; altitude = 1.25
    yaw = math.atan2(-normal[0], normal[1])
    stand = np.r_[contact-.55*normal, altitude]
    mission.metrics = dict(G=None, p_D=None, D=None)
    mission.action = 'Verify'; mission.status = '歧义开口 / 对准与标定'
    mission.reason = '累积测距支撑 / YOLO 语义提供候选；接触验证通行性'
    w.fly([stand], yaw=yaw, speed=.15)
    w.target = stand.copy(); w.stage = 'VERIFY_CALIBRATE'
    start = w.data.time
    while w.data.time-start < .65:
        w.step(yaw=yaw, speed=.10); sensors.measure_force(calibrate=True)
    observer = sensors.contact; threshold = observer.threshold
    w.stage = 'GLASS_VERIFY'; w.target = np.r_[contact+.16*normal, altitude]
    mission.status = '歧义开口 / 低速通行验证'
    start = float(w.data.time); peak = 0.; outcome = 'inconclusive'
    last_force_time = -1.; exceed = 0; stable = 0.; valid_samples = 0; trace = []
    failure = 'verification_timeout'; contact_position = None
    while w.data.time-start < cfg.verify_timeout:
        w.step(yaw=yaw, speed=cfg.verify_speed)
        measurement = sensors.measure_force()
        pos, vel, R, gyro = w.observed()
        fn = observer.normal_force(normal)
        fresh_force = measurement['valid'] and w.data.time-measurement['time_s'] < .04
        if measurement['time_s'] > last_force_time:
            last_force_time = measurement['time_s']; valid_samples += int(fresh_force)
            if fresh_force:
                peak = max(peak, fn)
                near = abs((pos[:2]-center)@normal) < .40 and abs((pos[:2]-contact)@tangent) < .25
                exceed = exceed+1 if near and fn > threshold else 0
            else: exceed = 0
            trace.append(dict(time_s=float(w.data.time), position_m=pos.tolist(), speed_m_s=float(np.linalg.norm(vel)),
                              normal_force_N=fn, threshold_N=threshold, force_valid=bool(fresh_force),
                              external_force_N=measurement['external_force_N']))
        if exceed >= 2:
            outcome = 'blocked'; failure = ''; contact_position = pos.tolist(); break
        if R[2, 2] < math.cos(math.radians(25)) or np.linalg.norm(vel) > .55:
            failure = 'verification_motion_limit'; break
        packet = sensors.packet
        forward_clear = False
        if packet is not None and w.data.time-packet.time < .25:
            ids = (packet.directions[:, :2]@normal > .85) & packet.valid
            forward_clear = ids.sum() >= 5 and float(np.mean(packet.distances[ids] > .45)) > .8
        completed = ((pos[:2]-center)@normal > .12 and np.linalg.norm(pos-w.target) < .065
                     and np.linalg.norm(vel) < .08 and forward_clear and fresh_force
                     and valid_samples >= 15 and len(observer.samples) >= 10)
        stable = stable+cfg.timestep if completed else 0.
        if stable > .20: outcome = 'free'; failure = ''; break
    event = dict(opening=rec.stable_id, start_time_s=start, end_time_s=float(w.data.time), outcome=outcome,
                 reason=failure, center_xy_m=center.tolist(), normal_xy=normal.tolist(),
                 force_peak_N=peak, threshold_N=threshold, valid_force_samples=valid_samples,
                 actual_contact_position=contact_position, commanded_speed_m_s=cfg.verify_speed,
                 calibration=sensors.contact.record(), trace=trace,
                 method='bilateral accumulated range + low-speed IMU-residual verification; thesis-inspired')
    mission.verifications.append(event)
    mission.memory.verified(rec.stable_id, float(w.data.time), outcome, peak,
                           semantic_preserved=rec.semantic, threshold_N=threshold,
                           valid_validation_position=outcome == 'free', completed=outcome != 'inconclusive',
                           reason=failure, force_source='calibrated IMU/AHRS/motor residual estimate')
    if outcome == 'free':
        mission.status = '验证净空 / 通行'
        exit_point = center + .75 * normal
        if free_exit is None:
            w.fly([np.r_[exit_point, altitude]], yaw=yaw)
        else:
            free_exit(rec, exit_point, yaw)
        mission.save_key(rec.stable_id+'_verified_free'); return outcome
    # Unload the frame through ordinary rotor control. Ground-truth contact is
    # not an input, including during recovery.
    mission.status = '验证受阻 / 退离恢复' if outcome == 'blocked' else '验证未完成 / 退离'
    w.stage = 'BACKOFF'; w.target = np.r_[contact-.85*normal, altitude]
    stable = 0.; start = w.data.time
    while w.data.time-start < 12:
        w.step(yaw=yaw, speed=.25); measurement = sensors.measure_force()
        pos, vel, R, gyro = w.observed()
        ok = (observer.normal_force(normal) < max(.25, threshold*.5) and np.linalg.norm(vel) < .12
              and R[2, 2] > .96 and np.linalg.norm(gyro) < .5 and (pos[:2]-center)@normal < -.70)
        stable = stable+cfg.timestep if ok else 0.
        if stable > .20: break
    else: raise PhysicsFailure('verification_recovery_timeout')
    if outcome == 'inconclusive': rec.bypass_selected = True
    mission.save_key(rec.stable_id+'_verification_'+outcome)
    return outcome
