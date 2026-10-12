"""Collect Panda joint-space datasets with MuJoCo and LQR trajectory tracking.

Every environment is the Panda with a box payload (random mass and position) attached to the
end-effector. Each run starts from a random joint configuration, tracks a random Fourier
excitation trajectory with a joint-space LQR at 50 Hz, and logs the resulting data.

Dependencies: numpy, scipy, mujoco and pinocchio.

Payload: a box body at `payload_pos` (end-effector frame), with mass `payload_mass`, its COM at the body
origin (`payload_com` = `payload_pos`) and isotropic inertia `payload_inertia` = 0.001 kg m^2 about the COM.
Only mass and position are random. All of them are saved with every run (the nominal env has mass 0).

Example (the 100 random environments of the CaDeLaC dataset, plus the robot-only one):
    python -m cadelac.control.lqr_dataset_collection --n_envs 100 --runs_per_env 20 --nominal

Random numbers come from numpy's global generator, seeded once with --seed. The payloads of all
`n_envs` environments are drawn first, then every run draws its initial pose and reference. To
reproduce the published data, keep --n_envs 100, --seed 0 and collect the environments in order.
"""
import argparse
import pickle
import time
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin
from scipy.linalg import solve_discrete_are

from cadelac.control.excitation_generator import generate_fourier_traj, obtain_valid_traj_param_simplified

ROBOT_DIR = Path(__file__).resolve().parent.parent / "robots" / "panda"
DATASET_DIR = Path(__file__).resolve().parent.parent / "learning" / "datasets" / "panda"
DEFAULT_SUFFIX = "_recreated"  # keeps the published datasets in the same folder untouched
ROBOT_XML = "panda_no_collision.xml"
SCENE_XML = "scene_no_collision.xml"
EE_BODY = '<body name="end_effector" pos="0 0 0.0584"/>'

CTRL_PERIOD = 0.02
VEL_LIM = np.array([2.0, 2.0, 2.0, 2.0, 2.3, 2.3, 2.3])
Q_INIT_CENTER = np.array([0, -0.4, 0, -2.4, 0, 2.2, -0.7853])
Q_INIT_LOW = np.array([-0.5, -0.5, -0.5, -0.1, -0.5, -0.7, -0.5])
Q_INIT_HIGH = np.array([0.5, 0.5, 0.5, 0.9, 0.5, 0.2, 0.5])
BOX_HALF_SIZE = 0.05
BOX_INERTIA = 0.001  # diagonal of the inertia about the COM (kg m^2), same for every payload
BOX_COM_OFFSET = np.zeros(3)  # COM relative to the box body origin


def sample_payloads(n_envs, seed):
    """Box position (n_envs, 3) and mass (n_envs,), drawn in the order used for the paper dataset."""
    np.random.seed(seed)
    pos = np.zeros((n_envs, 3))
    mass = np.zeros(n_envs)
    for i in range(n_envs):
        pos[i] = np.random.uniform(low=-0.3, high=0.3, size=3)
        mass[i] = np.random.uniform(low=0.0, high=4, size=1)[0]
    return pos, mass


def build_model(payload_pos=None, payload_mass=None):
    """Panda MuJoCo model, with a box payload at the end-effector if payload_pos is set."""
    robot_xml = (ROBOT_DIR / ROBOT_XML).read_text()
    robot_xml = robot_xml.replace('meshdir="assets"', f'meshdir="{(ROBOT_DIR / "assets").as_posix()}"')
    if payload_pos is not None:
        pos = " ".join(repr(float(x)) for x in payload_pos)
        size = " ".join([repr(BOX_HALF_SIZE)] * 3)
        inertia = " ".join([repr(BOX_INERTIA)] * 3)
        com = " ".join(repr(float(x)) for x in BOX_COM_OFFSET)
        box = (
            '<body name="end_effector" pos="0 0 0.0584">'
            f'<body name="ee_box" pos="{pos}">'
            f'<geom name="ee_box" class="visual" type="box" size="{size}" rgba="0.5 0.5 0.5 0.5"/>'
            f'<inertial pos="{com}" mass="{float(payload_mass)!r}" diaginertia="{inertia}"/>'
            "</body></body>"
        )
        assert EE_BODY in robot_xml
        robot_xml = robot_xml.replace(EE_BODY, box)
    scene_xml = (ROBOT_DIR / SCENE_XML).read_text()
    return mujoco.MjModel.from_xml_string(scene_xml, assets={ROBOT_XML: robot_xml.encode()})


def excitation_config(duration):
    """Fourier excitation trajectory settings and Panda joint limits."""
    upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    joints_range = upper - lower
    joints_range[0] = 2.0
    fourier_config = {"order": 5, "duration": duration}
    robot_config = {
        "njoints": 7,
        "upper_joint_pos_limits": upper,
        "lower_joint_pos_limits": lower,
        "joint_vel_limits": 2.0 * np.ones(7),
        "joints_range": joints_range,
    }
    return fourier_config, robot_config


class LQRCollector:
    """Simulates one Panda environment at a time, tracking an excitation reference with LQR."""

    def __init__(self, duration=10.0, hist_length=15, use_viewer=False):
        self.sim_length = int((duration + CTRL_PERIOD * hist_length) / CTRL_PERIOD)
        self.fourier_config, self.robot_config = excitation_config(duration + CTRL_PERIOD * hist_length)
        self.pin_model = pin.buildModelFromMJCF((ROBOT_DIR / ROBOT_XML).as_posix())
        self.pin_data = self.pin_model.createData()
        self.model = None
        self.use_viewer = use_viewer
        self.viewer = None

    def set_env(self, payload_pos=None, payload_mass=None, plain_torque_terms=False):
        """Set the simulated environment.

        The tau_m/c/g terms are computed with the simulated model, or with the plain Panda model (no box)
        if `plain_torque_terms` is set, as done for the nominal env of the published dataset.
        """
        self.model = build_model(payload_pos, payload_mass)
        self.data = mujoco.MjData(self.model)
        self.terms_model = build_model() if plain_torque_terms else self.model
        self.scratch = mujoco.MjData(self.terms_model)
        self.joint_ids = [self.model.joint(self.model.actuator(i).name.replace("actuator", "joint")).id
                          for i in range(self.model.nu)]
        assert self.joint_ids == list(range(7)) and self.model.nv == 7
        self.joint_ranges = self.model.jnt_range[self.joint_ids]

    def reset(self):
        """Random initial pose and excitation reference. Returns False if no valid reference was found."""
        q_init = Q_INIT_CENTER + np.random.uniform(low=Q_INIT_LOW, high=Q_INIT_HIGH, size=7)
        q_init = np.clip(q_init, self.joint_ranges[:, 0] + 0.1, self.joint_ranges[:, 1] - 0.1)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = q_init
        mujoco.mj_forward(self.model, self.data)

        robot_config = dict(self.robot_config, init_pos=q_init, init_vel=np.zeros(7))
        _, _, _, _, params = obtain_valid_traj_param_simplified(self.fourier_config, robot_config)
        if params is None:
            return False
        _, q, qd, _ = generate_fourier_traj(self.fourier_config["order"], self.fourier_config["duration"], 7,
                                            params, q_init, np.zeros(7), fps=1.0 / CTRL_PERIOD)
        self.q_ref, self.qd_ref = q.T, qd.T
        return True

    def lqr_torque(self, q_ref, qd_ref):
        m = self.model
        q, qd = self.data.qpos[self.joint_ids], self.data.qvel[self.joint_ids]
        Mq_inv = np.zeros((m.nv, m.nv))
        mujoco.mj_solveM(m, self.data, Mq_inv, np.eye(m.nv))

        # Double integrator x = (q, qd) with u = M^-1 tau (gravity and Coriolis are compensated)
        A = np.eye(14)
        A[:7, 7:] = CTRL_PERIOD * np.eye(7)
        B = np.zeros((14, 7))
        B[7:, :] = CTRL_PERIOD * Mq_inv
        Q = np.diag(100.0 * np.ones(14))
        R = 0.05 * np.eye(7)
        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

        x_error = np.concatenate((q_ref, qd_ref)) - np.concatenate((q, qd))
        return K @ x_error + self.data.qfrc_bias[self.joint_ids]

    def run(self):
        """One run. Returns the logged data, or None if the run was invalid and has to be discarded."""
        if not self.reset():
            return None
        if self.use_viewer:
            self.open_viewer()
        log = {k: [] for k in ("t", "qp", "qv", "qa", "tau", "qp_ref", "qv_ref")}
        for step in range(self.sim_length):
            q_ref, qd_ref = self.q_ref[:, step], self.qd_ref[:, step]
            qa = self.data.qacc[self.joint_ids]
            Mq = np.zeros((self.model.nv, self.model.nv))
            mujoco.mj_fullM(self.model, Mq, self.data.qM)

            log["t"].append(step * CTRL_PERIOD)
            log["qp"].append(self.data.qpos[self.joint_ids].copy())
            log["qv"].append(self.data.qvel[self.joint_ids].copy())
            log["qa"].append(qa.copy())
            log["tau"].append(Mq @ qa + self.data.qfrc_bias[self.joint_ids])
            log["qp_ref"].append(q_ref.copy())
            log["qv_ref"].append(qd_ref.copy())

            self.data.ctrl[:] = self.lqr_torque(q_ref, qd_ref)
            for _ in range(int(CTRL_PERIOD / self.model.opt.timestep)):
                mujoco.mj_step(self.model, self.data)
            if self.viewer is not None:
                self.viewer.sync()
                time.sleep(CTRL_PERIOD)

            if step + 1 >= self.sim_length:
                break
            if not self.is_healthy():
                return None
        return {k: np.array(v) for k, v in log.items()}

    def open_viewer(self):
        """(Re)open the MuJoCo viewer on the current run. The data object changes with every reset."""
        import mujoco.viewer
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def is_healthy(self):
        q, qd = self.data.qpos[self.joint_ids], self.data.qvel[self.joint_ids]
        return not (np.any(q < self.joint_ranges[:, 0]) or np.any(q > self.joint_ranges[:, 1])
                    or np.any(np.abs(qd) > VEL_LIM))

    def torque_terms(self, qp, qv, qa):
        """Inertial, Coriolis and gravity torques of the simulated model (with payload) at (qp, qv, qa)."""
        d = self.scratch
        d.qpos[:], d.qvel[:] = qp, qv
        mujoco.mj_forward(self.terms_model, d)
        Mq = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.terms_model, Mq, d.qM)
        tau_m, tau_cg = Mq @ qa, d.qfrc_bias.copy()
        d.qvel[:] = 0.0
        mujoco.mj_forward(self.terms_model, d)
        tau_g = d.qfrc_bias.copy()
        return tau_m, tau_cg - tau_g, tau_g

    def nominal_terms(self, qp, qv, qa):
        """Same terms for the robot without payload, from the Pinocchio model."""
        H = pin.crba(self.pin_model, self.pin_data, qp)
        H = np.triu(H) + np.triu(H, 1).T
        tau_g = pin.nonLinearEffects(self.pin_model, self.pin_data, qp, np.zeros_like(qv))
        tau_cg = pin.nonLinearEffects(self.pin_model, self.pin_data, qp, qv)
        return H @ qa, tau_cg - tau_g, tau_g

    def add_torque_keys(self, run):
        """tau_m/c/g of the simulated model and diff_tau* with respect to the nominal (no payload) model."""
        keys = ("tau_m", "tau_c", "tau_g", "diff_tau", "diff_tau_m", "diff_tau_c", "diff_tau_g")
        terms = {k: [] for k in keys}
        for qp, qv, qa, tau in zip(run["qp"], run["qv"], run["qa"], run["tau"]):
            sim_m, sim_c, sim_g = self.torque_terms(qp, qv, qa)
            nom_m, nom_c, nom_g = self.nominal_terms(qp, qv, qa)
            terms["tau_m"].append(sim_m)
            terms["tau_c"].append(sim_c)
            terms["tau_g"].append(sim_g)
            terms["diff_tau"].append(tau - (nom_m + nom_c + nom_g))
            terms["diff_tau_m"].append(sim_m - nom_m)
            terms["diff_tau_c"].append(sim_c - nom_c)
            terms["diff_tau_g"].append(sim_g - nom_g)
        run.update({k: np.array(v) for k, v in terms.items()})


def collect(n_envs=100, runs_per_env=20, env_ids=None, nominal=False, seed=0, duration=10.0, hist_length=15,
            viewer=False):
    """Collect `runs_per_env` valid runs for each env in `env_ids` (default all `n_envs`).

    With `nominal`, a last environment with a massless payload is added (env_id = n_envs). As in the
    published dataset, it is the robot with a zero-mass box at the end-effector (the box still adds its
    0.001 kg m^2 inertia), and its random stream is restarted from `seed`. Its tau_m/c/g terms are computed
    with the plain Panda model, as in the published dataset (the logged tau is from the simulated model).
    """
    env_ids = list(range(n_envs)) if env_ids is None else list(env_ids)
    payload_pos, payload_mass = sample_payloads(n_envs, seed)
    envs = [(i, payload_pos[i], payload_mass[i]) for i in env_ids]
    if nominal:
        envs.append((n_envs, np.zeros(3), 0.0))

    collector = LQRCollector(duration, hist_length, viewer)
    dataset = {k: [] for k in ("labels", "t", "qp", "qv", "qa", "env_id", "tau", "tau_m", "tau_c", "tau_g",
                               "diff_tau", "diff_tau_m", "diff_tau_c", "diff_tau_g", "qp_ref", "qv_ref",
                               "payload_pos", "payload_mass", "payload_com", "payload_inertia")}
    start = time.time()
    for env_id, pos, mass in envs:
        if nominal and env_id == n_envs:
            np.random.seed(seed)  # the nominal env was collected separately, with a fresh random stream
        collector.set_env(pos, mass, plain_torque_terms=nominal and env_id == n_envs)
        for run_id in range(runs_per_env):
            run = None
            while run is None:
                run = collector.run()
            collector.add_torque_keys(run)
            run["env_id"] = np.full(len(run["t"]), env_id)
            run["payload_pos"] = pos
            run["payload_mass"] = np.array([mass])
            run["payload_com"] = pos + BOX_COM_OFFSET
            run["payload_inertia"] = np.full(3, BOX_INERTIA)
            dataset["labels"].append(f"env_{env_id}_{run_id}")
            for key, value in run.items():
                dataset[key].append(value)
        # The original collection script prepared a new run after every run, including the last one of an
        # environment, which consumed random numbers. Keep doing so to reproduce the published data.
        collector.reset()
        print(f"env {env_id} done | {time.time() - start:.0f}s", flush=True)
    return dataset


def parse_ids(text):
    """'0-4' -> [0, 1, 2, 3, 4], '0,3,5' -> [0, 3, 5], '' -> [] (e.g. only the --nominal env)."""
    ids = []
    for part in filter(None, text.split(",")):
        lo, _, hi = part.partition("-")
        ids += list(range(int(lo), int(hi or lo) + 1))
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_envs", type=int, default=100, help="number of random payloads drawn")
    parser.add_argument("--env_ids", type=parse_ids, default=None, help="environments to run, e.g. 0-4 (default: all)")
    parser.add_argument("--runs_per_env", type=int, default=20)
    parser.add_argument("--nominal", action="store_true", help="add a last environment without payload")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=float, default=10.0, help="run length in seconds, without the history")
    parser.add_argument("--hist_length", type=int, default=15, help="extra samples for the model history")
    parser.add_argument("--viewer", action="store_true", help="show each run in the MuJoCo viewer (real time)")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"output file (default: learning/datasets/panda/panda_mj_<envs>_rand_envs_<runs>_runs_50Hz_lqr{DEFAULT_SUFFIX}.pkl)")
    parser.add_argument("--overwrite", action="store_true", help="replace the output file if it exists")
    args = parser.parse_args()

    if args.out is None:
        env_ids = range(args.n_envs) if args.env_ids is None else args.env_ids
        name = f"panda_mj_{len(env_ids) + args.nominal}_rand_envs_{args.runs_per_env}_runs_50Hz_lqr"
        if args.env_ids:
            name += "_env_ids_" + "_".join(map(str, args.env_ids))
        elif args.env_ids is not None:
            name = f"panda_mj_nominal_env_{args.runs_per_env}_runs_50Hz_lqr"
        if args.seed != 0:
            name += f"_seed_{args.seed}"
        args.out = DATASET_DIR / f"{name}{DEFAULT_SUFFIX}.pkl"
    if args.out.exists() and not args.overwrite:
        parser.error(f"{args.out} already exists, use --overwrite or --out")

    dataset = collect(args.n_envs, args.runs_per_env, args.env_ids, args.nominal, args.seed,
                      args.duration, args.hist_length, args.viewer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(dataset, f)
    print(f"saved {len(dataset['labels'])} runs to {args.out}")


if __name__ == "__main__":
    main()
