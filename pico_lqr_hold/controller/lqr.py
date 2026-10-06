"""
LQR (with integral action) design for the Swift Pico in drone.xml.

Everything physical -- mass, gravity, inertia -- is read off the MuJoCo model,
so if you edit drone.xml the gains follow automatically.

State (15), all expressed in the drone's HEADING frame (world axes rotated by
the current yaw), as ERRORS from the setpoint:

    x = [ ex  ey  ez  |  vx  vy  vz  |  phi  theta  psi  |  p  q  r  |  Ix  Iy  Iz ]
          position       velocity       roll pitch yaw      body rates   integral of position error

Input (4):

    u = [ dF  tau_x  tau_y  tau_z ]     dF = total thrust minus hover thrust (N),
                                        tau_* = body torques (N*m)

Linearised about hover (small roll/pitch):

    ex''   =  g * theta            (pitching nose-down pushes you along +x)
    ey''   = -g * phi              (rolling right pushes you along -y ... see sign
                                    check in the README / run_sim output)
    ez''   =  dF / m
    phi''  =  tau_x / Ixx,  theta'' = tau_y / Iyy,  psi'' = tau_z / Izz
    I'     =  e                    (integral states -> zero steady-state error)

The plant is discretised with a zero-order hold at the simulator timestep
(MuJoCo holds data.ctrl constant over each mj_step, so ZOH is exact) and the
infinite-horizon discrete Riccati equation is solved with SciPy:

    P = A'PA - A'PB (R + B'PB)^-1 B'PA + Q       K = (R + B'PB)^-1 B'PA
    u = -K x
"""
import numpy as np
import mujoco
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete

# Order of the state vector -- shared with controller.py
IDX_POS = slice(0, 3)
IDX_VEL = slice(3, 6)
IDX_ATT = slice(6, 9)
IDX_RATE = slice(9, 12)
IDX_INT = slice(12, 15)
N_STATE, N_INPUT = 15, 4


def read_physical_params(model, body_name="drone"):
    """Total mass, gravity and body-frame inertia (Ixx, Iyy, Izz) about the
    whole-drone centre of mass, including the four small propeller bodies."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    mass = float(model.body_subtreemass[bid])
    g = float(-model.opt.gravity[2])
    com = data.subtree_com[bid].copy()

    # Sum each body's inertia (rotated to world) + parallel-axis term about the COM
    I_world = np.zeros((3, 3))
    tree = [b for b in range(model.nbody) if model.body_rootid[b] == bid]   # drone + prop bodies
    for b in tree:
        m_b = model.body_mass[b]
        if m_b <= 0:
            continue
        Ri = data.ximat[b].reshape(3, 3)
        I_world += Ri @ np.diag(model.body_inertia[b]) @ Ri.T
        r = data.xipos[b] - com
        I_world += m_b * (r @ r * np.eye(3) - np.outer(r, r))

    Rb = data.xmat[bid].reshape(3, 3)
    I_body = Rb.T @ I_world @ Rb
    return mass, g, np.diag(I_body).copy()


def build_linear_model(mass, g, inertia):
    """Continuous-time (A, B) of the 15-state error model described above."""
    Ixx, Iyy, Izz = inertia
    A = np.zeros((N_STATE, N_STATE))
    B = np.zeros((N_STATE, N_INPUT))

    # position' = velocity
    A[0:3, 3:6] = np.eye(3)
    # velocity' : tilt -> horizontal acceleration
    A[3, 7] = g       # ex''  =  g * theta
    A[4, 6] = -g      # ey''  = -g * phi
    B[5, 0] = 1.0 / mass
    # attitude' = body rates (small-angle)
    A[6:9, 9:12] = np.eye(3)
    B[9, 1] = 1.0 / Ixx
    B[10, 2] = 1.0 / Iyy
    B[11, 3] = 1.0 / Izz
    # integral' = position error
    A[12:15, 0:3] = np.eye(3)
    return A, B


def default_weights():
    """Bryson's-rule style weights: Q_ii = 1 / (acceptable error_i)^2,
    R_ii = 1 / (acceptable effort_i)^2.  Smaller 'acceptable' = tighter."""
    q = np.zeros(N_STATE)
    q[0:2] = 1 / 0.50**2     # x, y error           (m)
    q[2]   = 1 / 0.50**2     # z error              (m)
    q[3:5] = 1 / 0.80**2     # x, y velocity        (m/s)
    q[5]   = 1 / 0.80**2     # z velocity           (m/s)
    q[6:8] = 1 / 0.15**2     # roll, pitch          (rad)
    q[8]   = 1 / 0.20**2     # yaw                  (rad)
    q[9:11] = 1 / 1.00**2    # roll, pitch rate     (rad/s)
    q[11]  = 1 / 1.00**2     # yaw rate             (rad/s)
    q[12:15] = 1 / 0.60**2   # integral of position error (m*s)

    r = np.array([1 / 6.0**2,    # dF      (N)
                  1 / 0.8**2,    # tau_x   (N*m)
                  1 / 0.8**2,    # tau_y   (N*m)
                  1 / 0.3**2])   # tau_z   (N*m)
    return np.diag(q), np.diag(r)


def design_lqr(model, dt=None, Q=None, R=None):
    """Return (K, info). K is 4x15, u = -K @ x."""
    mass, g, inertia = read_physical_params(model)
    dt = model.opt.timestep if dt is None else dt
    A, B = build_linear_model(mass, g, inertia)
    Qd, Rd = default_weights()
    Q = Qd if Q is None else Q
    R = Rd if R is None else R

    Ad, Bd, _, _, _ = cont2discrete((A, B, np.eye(N_STATE), np.zeros((N_STATE, N_INPUT))), dt, method="zoh")
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)

    cl_poles = np.linalg.eigvals(Ad - Bd @ K)
    info = dict(mass=mass, g=g, inertia=inertia, A=A, B=B, Ad=Ad, Bd=Bd, P=P,
                cl_poles_z=cl_poles,
                cl_poles_s=np.log(cl_poles.astype(complex)) / dt)
    return K, info


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    m = mujoco.MjModel.from_xml_path(os.path.join(here, "..", "model", "drone.xml"))
    K, info = design_lqr(m)
    np.set_printoptions(precision=3, suppress=True, linewidth=160)
    print(f"mass = {info['mass']:.4f} kg, g = {info['g']:.2f}, inertia = {info['inertia']}")
    labels = "ex ey ez vx vy vz phi theta psi p q r Ix Iy Iz".split()
    print("\nK (rows: dF, tau_x, tau_y, tau_z):")
    print("      " + " ".join(f"{l:>7s}" for l in labels))
    for name, row in zip(["dF", "tau_x", "tau_y", "tau_z"], K):
        print(f"{name:>6s}" + " ".join(f"{v:7.3f}" for v in row))
    # controllability sanity check
    n = N_STATE
    Ad, Bd = info["Ad"], info["Bd"]
    C = np.hstack([np.linalg.matrix_power(Ad, i) @ Bd for i in range(n)])
    print(f"\ncontrollability rank = {np.linalg.matrix_rank(C)} / {n}")
    s = info["cl_poles_s"]
    print("closed-loop poles (s-plane, rad/s), slowest 6:")
    print(s[np.argsort(-s.real)][:6])
