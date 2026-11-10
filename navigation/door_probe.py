"""Real MuJoCo contact/hinge instrumentation for one existing rotor Probe.

Ideal hinge encoder + contact wrench instrumentation is explicit in this
experiment. No inertia/friction is exposed. Force is geom1 -> geom2;
contact.frame stores axes as ROWS, hence world = frame.T @ local.
https://mujoco.readthedocs.io/en/latest/programming/simulation.html#contacts
"""
import mujoco
import numpy as np
from .door_dynamics import ProbeTrace
from .interactions import probe


class DoorInstrument:
    def __init__(self, world, opening):
        self.world = world
        m = world.model
        self.jid = m.joint('hinge_' + opening['id']).id
        self.qa = int(m.jnt_qposadr[self.jid])
        self.da = int(m.jnt_dofadr[self.jid])
        panel_id = m.body('panel_' + opening['id']).id
        self.geoms = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == panel_id}

    def state(self):
        d = self.world.data
        return float(d.qpos[self.qa]), float(d.qvel[self.da])

    def contact(self):
        w = self.world
        m, d = w.model, w.data
        pivot, axis = d.xanchor[self.jid].copy(), d.xaxis[self.jid].copy()
        torque = 0.
        contacts = []
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = int(c.geom1), int(c.geom2)
            if not ((a in self.geoms and b in w.robot_geoms) or
                    (b in self.geoms and a in w.robot_geoms)):
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, wrench)
            sign = 1. if b in self.geoms else -1.
            force = sign * c.frame.reshape(3, 3).T @ wrench[:3]
            point = c.pos.copy()
            moment = float(np.cross(point-pivot, force) @ axis)
            torque += moment
            contacts.append(dict(point_world=point.tolist(), force_on_door_world=force.tolist(),
                                 torque_about_hinge=moment, door_is_geom2=bool(b in self.geoms)))
        return torque, contacts


def record_probe(world, opening, instrument, progress=None):
    """Use the unchanged probe() controller; sample before mj_forward/callback.

    mj_step2 integrates state using the contacts of that interval. The optional
    after-step hook reads that solved wrench BEFORE the 20 Hz mj_forward clears
    or recomputes it. Assign the wrench to the preceding state/time interval.
    """
    rows = [[float(world.data.time), *instrument.state()]]
    torques, contact_debug = [], []
    previous = world.after_step_observer

    def sample(w):
        if previous:
            previous(w)
        tau, contacts = instrument.contact()
        torques.append(tau)
        rows.append([float(w.data.time), *instrument.state()])
        contact_debug.append(contacts)

    world.after_step_observer = sample
    try:
        legacy_z, _ = probe(world, opening, instrument.state, progress=progress)
    finally:
        world.after_step_observer = previous
    a = np.asarray(rows)
    trace = ProbeTrace(a[:, 0]-a[0, 0], np.r_[torques, torques[-1]], a[:, 1], a[:, 2])
    trace.validate()
    return trace, contact_debug, legacy_z
