import numpy as np
from scipy.stats import truncnorm
import pinocchio as pin

def generate_constraints(order: int):
    """
    Trajectories are parameterized as
        q = \sum_k A_k*1/wk sin(k*wt) - B_k*1/wk cos(k*wt)
        qd = \sum_k A_k cos(k*wt) + B_k sin(k*wt)
        qdd = -\sum_k A_k*wk sin(k*wt) + B_k*wk cos(k*wt)

    Initial position, velocity, acceleration constraints: q(0) = qd(0) = qdd(0) = 0
        cA:  sum_k A_1 + ... + A_k + ... + A_L = 0
        cB1: sum_k B_1/1 + ... B_k/k + ... + B_L/L = 0
        cB2: sum_k B_1*1 + ... B_k*k + ... + B_L*L = 0
        where l is the order, so for B, to simplify the
        constraints, we can rewrite the constraints
        [
            [1, 1/2, 1/3, ..., 1/l],
            [1, 2, 3, ..., l]
        ] * B = 0, where B = [B_1, B_2, ..., B_L]
        to the reduced echelon_form so that we can generate
        parameters that satisfy the constraints

    Param:
        order: int, order of the Fourier series
    Return:
        cA: constraints vector, A[0] = -np.dot(cA,A)
        cB: constraints matrix, B[1] = -np.dot(cB[0][2:], B[2:])
                                B[0] = -np.dot(cB[1][1:], B[1:])
    """
    cA = np.ones(order)
    cB = np.array(
        [[1 / i for i in range(1, order + 1)], [i for i in range(1, order + 1)]]
    )
    cB_reduced_echelon_form = np.array([(cB[0] - cB[1]) / (cB[0] - cB[1])[1], cB[1]])
    return cA, cB_reduced_echelon_form

def generate_random_param(order: int, njoints: int, param_range = None):
    """
    Generate constant terms for the Fourier basis, each joint has
    different frequencies

    Param:
        order, njoints
    Return:
        params: an array that includes parameters, shape (2, order, njoints)
            A: parameters for the sin basis, shape (order, njoints)
            B: parameters for the cos basis, shape (order, njoints)
    """
    if param_range is None:
        params = (
            np.random.randn(2, order, njoints) * 0.02 # 0.02 works for duration 100, 0.04 works for duration (40, 80], 0.08 works for duration <= 40
                                                    # for single joint, duration 100, 0.08 works
        )  # use tiny variance to tweak the params
    else:
        rand_comp = np.random.uniform(low=-1.0, high=1.0, size=(2, order, njoints))
        param_norm = 1.0 * (param_range * 0.5 * 1.0 / order)
        # print(f'param_norm {param_norm}')
        params = rand_comp * param_norm

    cA, cB = generate_constraints(order)
    A, B = params[0], params[1]
    A[-1] = -np.dot(cA[:-1], A[:-1])
    B[1] = -np.dot(cB[0][2:], B[2:])
    B[0] = -np.dot(cB[1][1:], B[1:])
    params = np.array([A, B])
    return params


def generate_truncated_param(
    order: int, njoints: int, fourier_config: dict, robot_config: dict
):
    """
    Generate parameters from a truncated gaussian distribution which enforces
    the position and velocities constraints, constraints refer to (14d)-(14e) in
    the paper:
        Parameter Identification of the UA LBR iiwa Robot Including Constraints on
        Physical Feasibility
    """
    # TODO: still have bugs need to be fixed
    duration = fourier_config["duration"]
    oemga_f = 2 * np.pi / duration
    q_max, q_min, qd_max = (
        robot_config["upper_joint_pos_limits"],
        robot_config["lower_joint_pos_limits"],
        robot_config["joint_vel_limits"],
    )
    q_max, q_min, qd_max = np.array(q_max), np.array(q_min), np.array(qd_max)
    pos_constant = np.sum([1 / i * np.sqrt(2) for i in range(1, order + 1)])
    pos_upper1 = oemga_f * q_max / pos_constant
    pos_upper2 = -oemga_f * q_min / pos_constant

    vel_constant = order * np.sqrt(2)
    vel_upper = qd_max / vel_constant

    a_max = np.minimum(vel_upper, np.minimum(pos_upper1, pos_upper2))
    a_min = -a_max
    param_flat_shape = 2 * order

    params = []
    for i in range(njoints):
        param = truncnorm.rvs(a_min[i], a_max[i], size=param_flat_shape)
        params.append(param)

    params = np.array(params).T
    params = np.array(params).reshape(2, order, njoints)
    cA, cB = generate_constraints(order)
    A, B = params[0], params[1]
    A[-1] = -np.dot(cA[:-1], A[:-1])
    B[1] = -np.dot(cB[0][2:], B[2:])
    B[0] = -np.dot(cB[1][1:], B[1:])
    params = np.array([A, B])
    return params


def generate_identical_param(order: int, njoints: int):
    """
    Generate constant terms for Fourier basis, each joint has the same
    frequencies
    """
    params = np.random.randn(2, order)
    A_j, B_j = params[0].reshape(-1, 1), params[1].reshape(-1, 1)
    A = np.hstack([A_j for i in range(njoints)])
    B = np.hstack([B_j for i in range(njoints)])

    cA, cB = generate_constraints(order)
    A[-1] = -np.dot(cA[:-1], A[:-1])
    B[1] = -np.dot(cB[0][2:], B[2:])
    B[0] = -np.dot(cB[1][1:], B[1:])
    params = np.array([A, B])
    return params


def generate_fourier_traj(
    order, duration, njoints, params, init_pos=None, init_vel=None, fps=100
):
    if init_pos is None:
        init_pos = np.zeros(njoints)
    if init_vel is None:
        init_vel = np.zeros(njoints)
    step_size = 1 / fps

    omega_f = 2 * np.pi / duration
    num_samples = int(duration / step_size)
    # print(f'duration {duration} | num_samples {num_samples} | step_size {step_size}')
    t = np.linspace(0, duration, num_samples+1)
    q = np.zeros((t.shape[0], njoints)) + init_pos
    dq = np.zeros_like(q) + init_vel
    ddq = np.zeros_like(q)
    A, B = params[0], params[1]

    # the outer product of sin / cos and parameters will generate the trajectory as
    # [num_samples, njoints] shape
    for k in range(1, order + 1):
        q += np.outer(np.sin(omega_f * k * t), A[k - 1] / (omega_f * k)) - np.outer(
            np.cos(omega_f * k * t), B[k - 1] / (omega_f * k)
        )
        dq += np.outer(np.cos(omega_f * k * t), A[k - 1]) + np.outer(
            np.sin(omega_f * k * t), B[k - 1]
        )
        ddq += -np.outer(np.sin(omega_f * k * t), A[k - 1] * (omega_f * k)) + np.outer(
            np.cos(omega_f * k * t), B[k - 1] * (omega_f * k)
        )
    return t, q, dq, ddq

def is_traj_valid(q, dq, ddq, robot_config):
    upperPosLimit, lowerPosLimit, velLimit = (
        robot_config["upper_joint_pos_limits"],
        robot_config["lower_joint_pos_limits"],
        robot_config["joint_vel_limits"],
    )
    for q_i, dq_i, ddq_i in zip(q, dq, ddq):
        for j in range(len(q_i)):
            if q_i[j] > upperPosLimit[j]:
                # print(q_i[j], upperPosLimit[j])
                # print(f"q upper {j}")
                return False
            elif q_i[j] < lowerPosLimit[j]:
                # print(q_i[j], lowerPosLimit[j])
                # print(f"q lower {j}")
                return False
        if np.any(np.abs(dq_i) - velLimit > 0):
            # print(dq_i, "DQI ==========")
            # print(velLimit, "velLimit ===========")
            # print("dq")
            return False
    return True

def is_params_valid(params, fourier_config, robot_config):
    order = fourier_config["order"]
    duration = fourier_config["duration"]
    njoints = robot_config["njoints"]
    upperPosLimit, lowerPosLimit, velLimit = (
        robot_config["upper_joint_pos_limits"],
        robot_config["lower_joint_pos_limits"],
        robot_config["joint_vel_limits"],
    )
    upperPosLimit, lowerPosLimit, velLimit = (
        np.array(upperPosLimit),
        np.array(lowerPosLimit),
        np.array(velLimit),
    )
    start_idx = len(params) // 2
    # nonlinear constraints for positions and velocities
    extreme_constraint_mat = np.zeros((njoints, order * njoints))
    for i in range(njoints):
        extreme_constraint_mat[i, i * order : (i + 1) * order] = np.ones(order)

    def extreme_vel_func(x):
        A = x[:start_idx]
        B = x[start_idx:]
        A2 = A**2
        B2 = B**2
        A2B2 = A2 + B2
        root_A2B2 = np.sqrt(A2B2)
        return np.dot(extreme_constraint_mat, root_A2B2)

    extreme_vel_lb = np.inf
    extreme_vel_ub = velLimit
    for i in range(len(extreme_vel_ub)):
        if extreme_vel_ub[i] > 10000:
            extreme_vel_ub[i] = 10000
    extreme_vel_constraint_func = lambda x: -(extreme_vel_func(x) - extreme_vel_ub)
    if np.any(extreme_vel_constraint_func(params) < 0):
        print("VEL OUT OF BOUND")
        return False

    omega_f = 2 * np.pi / duration

    def extreme_pos_func(x):
        A = x[:start_idx]
        B = x[start_idx:]
        A2 = A**2
        B2 = B**2
        A2B2 = A2 + B2
        root_A2B2 = np.sqrt(A2B2)
        divider = np.array([1 / i for i in range(1, order + 1)] * njoints)
        root_A2B2_divided = root_A2B2 * divider
        return np.dot(extreme_constraint_mat, root_A2B2_divided)

    extreme_pos_lb = np.inf
    extreme_pos_ub = upperPosLimit * omega_f
    print(extreme_pos_ub)
    extreme_pos_constraint_func = lambda x: -(extreme_pos_func(x) - extreme_pos_ub)
    if np.any(extreme_pos_constraint_func(params) < 0):
        print("POS OUT OF BOUND")
        return False
    return True


def obtain_fourier_traj(params, fourier_config, robot_config):
    order = fourier_config["order"]
    duration = fourier_config["duration"]
    njoints = robot_config["njoints"]
    init_pos = robot_config["init_pos"]
    init_vel = robot_config["init_vel"]
    t, q, dq, ddq = generate_fourier_traj(
        order, duration, njoints, params, init_pos, init_vel
    )
    return t, q, dq, ddq


def obtain_valid_traj_param(fourier_config, robot_config):
    order = fourier_config["order"]
    njoints = robot_config["njoints"]
    valid_traj = False
    valid_param = False
    no_collision = False
    while not valid_traj or not valid_param or not no_collision:
        params = generate_random_param(order, njoints)
        t, q, dq, ddq = obtain_fourier_traj(params, fourier_config, robot_config)
        valid_traj = is_traj_valid(q, dq, ddq, robot_config)
        no_collision, ee_pos = test_traj(q, dq, ddq)

        _params = np.transpose(params, (0, 2, 1))
        _params = _params.flatten()
        valid_param = is_params_valid(_params, fourier_config, robot_config)
    return t, q, dq, ddq, params

def obtain_valid_traj_param_simplified(fourier_config, robot_config, max_try = 10_000):
    order = fourier_config["order"]
    njoints = robot_config["njoints"]
    param_range = robot_config['joints_range']
    valid_traj = False
    try_traj = 0
    while not valid_traj and (try_traj < max_try):
        params = generate_random_param(order, njoints, param_range)
        # print(f'try_traj {try_traj}')
        t, q, dq, ddq = obtain_fourier_traj(params, fourier_config, robot_config)
        valid_traj = is_traj_valid(q, dq, ddq, robot_config)
        _params = np.transpose(params, (0, 2, 1))
        _params = _params.flatten()
        try_traj = try_traj + 1

    if valid_traj:
        return t, q, dq, ddq, params
    else:
        print(f'Reached max number of sampled trajectory {max_try}')
        params = None
        return t, q, dq, ddq, params


def test_traj(model_pin, data_pin, q, dq, ddq):

    ee_id = model_pin.getFrameId("end_effector")

    ee_pos = []
    for q_i, dq_i, ddq_i in zip(q, dq, ddq):
        pin.forwardKinematics(model_pin, data_pin, q_i)

        if np.any((np.abs(dq_i) - model_pin.velocityLimit) > 0):
            return False, None
        if np.any((np.abs(q_i) - model_pin.upperPositionLimit) > 0):
            return False, None

        ee_pos_i = data_pin.oMi[ee_id].translation

        if ee_pos_i[0] < -0.5 or abs(ee_pos_i[1]) > 0.8 or ee_pos_i[2] < 0.2:
            return False, None

        ee_pos.append(ee_pos_i)

    return True, ee_pos