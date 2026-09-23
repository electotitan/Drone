"""
Makes the propeller meshes actually spin.

drone.xml gives each propeller its own body (prop_m1..prop_m4) with a free,
frictionless hinge joint (spin_m1..spin_m4, damping=0 armature=0
frictionloss=0) about the body's local +Z -- see the "Spinning propeller
bodies" comment above them in drone.xml. Nothing else in the model touches
those joints: the <actuator> motors apply thrust/yaw torque at sites on the
main drone body, not at these joints. So with no driver, qvel/qpos on all
four sit at exactly 0 for the whole sim -- the propellers never turn.

PropSpinner is that missing driver. Call apply() once per physics step,
right before mj_step(), with the same 4 motor forces you're about to write
to data.ctrl. It converts each force back to the motor's angular speed via
the model's own thrust law (F = Kt * omega^2, see drone.xml's ctrlrange
comment: Kt * 800^2 ~= 5.47 N) and writes that straight into the spin
joint's qvel, signed for that motor's rotation direction. The propeller
bodies are ~0.006 kg each (vs. 1.5 kg for the drone body), so overwriting
their velocity every step has no measurable effect on the flight dynamics
-- it's a purely visual/telemetry channel riding on top of the real
control loop in controller.py.
"""
import numpy as np
import mujoco

KT = 8.54858e-06  # N / (rad/s)^2 -- matches drone.xml: Kt * omega_max^2 =
                   # 8.54858e-06 * 800^2 ~= 5.47 N (the actuator ctrlrange)

# Spin direction about body +Z, from drone.xml's own comment:
#   "ccw motors (m1,m2) spin +Z, cw motors (m3,m4) spin -Z"
_JOINTS = ["spin_m1", "spin_m2", "spin_m3", "spin_m4"]
_DIRECTION = np.array([+1.0, +1.0, -1.0, -1.0])

# Below this thrust the true omega would be tiny/zero (motor idling or off).
# A little idle spin reads as "running" rather than "stopped" without being
# fast enough to look wrong when the drone is sitting on the ground pre-arm.
IDLE_OMEGA = 15.0  # rad/s


class PropSpinner:
    def __init__(self, model):
        self._dof = np.empty(4, dtype=int)
        for i, jname in enumerate(_JOINTS):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid == -1:
                raise ValueError(
                    f"drone.xml has no joint named '{jname}' -- PropSpinner "
                    f"needs the spin_m1..spin_m4 hinges on the propeller bodies")
            self._dof[i] = model.jnt_dofadr[jid]

    def apply(self, data, motor_forces):
        """motor_forces: length-4 [F1, F2, F3, F4] in N, same order/values
        you're about to assign to data.ctrl. Call before mj_step()."""
        omega = np.sqrt(np.clip(motor_forces, 0.0, None) / KT)
        omega = np.maximum(omega, IDLE_OMEGA)
        data.qvel[self._dof] = _DIRECTION * omega
