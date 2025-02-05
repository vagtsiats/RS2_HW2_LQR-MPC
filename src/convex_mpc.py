import linearization_lqr as lqr
import modeling as md
import numpy as np
import qpsolvers


def qp_matrices(Nh, Q, R, A, B, Pinf, max_t):
    # cost matrix
    size = (Nh - 1) * (M + N)
    H = np.zeros((size, size))
    H[:M, :M] = R
    H[-N:, -N:] = Pinf  # We put Pinf instead of QN here

    for k in range(Nh - 2):
        sidx = M + k * (M + N)
        eidx = sidx + N
        H[sidx:eidx, sidx:eidx] = Q

        sidx = M + k * (M + N) + N
        eidx = sidx + M
        H[sidx:eidx, sidx:eidx] = R

    # Cost vector
    h = np.zeros((size, 1))

    # Equality Constraints (dynamics)
    G = np.zeros(((Nh - 1) * N, size))

    G[:N, :M] = B
    G[:N, M : M + N] = -np.eye(N)
    for k in range(Nh - 2):
        sidx = (k + 1) * N
        sidx2 = k * (N + M) + M

        G[sidx : sidx + N, sidx2 : sidx2 + N] = A
        G[sidx : sidx + N, sidx2 + N : sidx2 + N + M] = B
        G[sidx : sidx + N, sidx2 + N + M : sidx2 + N + M + N] = -np.eye(N)

    d = np.zeros(((Nh - 1) * N, 1))

    # Inequality Constraints (force limits)
    lb = -np.ones((size, 1)) * np.inf
    ub = np.ones((size, 1)) * np.inf

    for k in range(Nh - 1):
        idx = k * (N + M)
        lb[idx : idx + M] = -max_t - u_bar
        ub[idx : idx + M] = max_t - u_bar

    return H, h, G, d, lb, ub


def mpc_controller(qp: qpsolvers.problem.Problem, x_current, x_ref, x_bar, u_bar):

    _, h, _, _, _, d, _, _ = qp.unpack()

    d[:N] = (-A @ (x_current - x_bar)).reshape((4,))
    for k in range(Nh - 2):
        h[k * (N + M) + M : k * (N + M) + M + N] = ((x_bar - x_ref).T @ Q).reshape((N,))
    h[-N:] = ((x_bar - x_ref).T @ Pinf).reshape((N,))

    qp.q = h
    qp.b = d

    qp_solution = qpsolvers.solve_problem(qp, solver="osqp")

    return qp_solution.x[:M].reshape((M, -1))


if __name__ == "__main__":

    N = 4  # state dim
    M = 2  # control dim

    # linearization states
    x_bar = np.array([[np.pi, 0.0, 0.0, 0.0]]).T
    u_bar = np.array([[0.0, 0.0]]).T

    max_torque = 10
    x_ref = np.array([[np.pi, 0.0, 0.0, 0.0]]).T
    x = np.array([[0.0, 0.0, 0.0, 0.0]]).T  # initial state

    MPC = False
    cov = 10e-4  # noise covariance

    img_name = None
    # img_name = "mpc/mpc_far"

    Nh = 30  # MPC lookahead horizon
    Q = 100 * np.eye(N)  # state cost
    R = 0.001 * np.eye(M)  # control cost
    QN = np.eye(N)

    A, B = lqr.linearize(x_bar, u_bar)
    Pinf, Kinf = lqr.infinite_lqr(A, B, QN, Q, R)

    H, h, G, d, lb, ub = qp_matrices(Nh=Nh, Q=Q, R=R, A=A, B=B, Pinf=Pinf, max_t=max_torque)

    qp_problem = qpsolvers.problem.Problem(P=H, q=h, A=G, b=d, lb=lb, ub=ub)

    # simulation
    T = 10
    t = 0

    states = x
    controls = np.copy(u_bar)
    u_max = np.ones_like(u_bar) * max_torque
    while t < T:

        noise = np.random.normal(0, cov, (4, 1))
        noise[2:, :] = 0
        x = x + noise

        # LQR
        u = u_bar - Kinf @ (x - x_ref)
        if MPC:
            u = u_bar + mpc_controller(qp_problem, x, x_ref, x_bar, u_bar)

        u_new = np.maximum(np.minimum(u, u_max), -u_max)

        x = md.rk4(x, u_new)

        states = np.hstack((states, x))
        controls = np.hstack((controls, u_new))
        t += md.DT

    md.visualize(states, controls=controls, ref_state=x_ref, name=img_name)
    md.animate(states, x_ref)
