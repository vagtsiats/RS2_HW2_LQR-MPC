import modeling as md
import jax.numpy as jnp
from jax import jacfwd, jit
import numpy as np

# from jax import config
# config.update("jax_enable_x64", True)


def linearize(x_bar, u_bar):

    A = jit(jacfwd(md.rk4, 0))(x_bar, u_bar).reshape(4, 4)
    B = jit(jacfwd(md.rk4, 1))(x_bar, u_bar).reshape(4, 2)

    return A, B


def infinite_lqr(A, B, Qn, Q, R, K=5000):

    # Init
    Ps = [np.zeros((4, 4))] * K
    Ks = [np.zeros((2, 4))] * (K - 1)

    # Riccati backward
    Ps[K - 1] = Qn
    for k in range(K - 2, -1, -1):
        tmp1 = R + B.T @ Ps[k + 1] @ B
        tmp2 = B.T @ Ps[k + 1] @ A
        Ks[k] = np.linalg.solve(tmp1, tmp2)
        ## From DP
        tmp = A - B @ Ks[k]
        Ps[k] = Q + Ks[k].T @ R @ Ks[k] + tmp.T @ Ps[k + 1] @ tmp
        ## End from DP

    # Let's take out the results we need
    Kinf = Ks[0]
    Pinf = Ps[0]

    return Pinf, Kinf


if __name__ == "__main__":

    x_bar = np.array([[np.pi, 0.0, 0.0, 0.0]]).T
    u_bar = np.array([[0.0, 0.0]]).T
    u_min = np.ones_like(u_bar) * (-10)
    u_max = np.ones_like(u_bar) * (10)

    Q = 1 * np.eye(4)
    R = 0.1 * np.eye(2)
    QN = np.eye(4)

    A, B = linearize(x_bar, u_bar)

    P, K = infinite_lqr(A, B, QN, Q, R)

    # print(linearize(x_bar, u_bar)[1])
    # print(P, K)

    # simulation
    T = 5
    t = 0

    x = np.copy(x_bar)
    states = x
    controls = np.copy(u_bar)

    x_ref = np.array([[0.0, np.pi / 2, 0, 0]]).T

    while t < T:

        u = u_bar - K @ (x - x_ref)

        u = np.maximum(np.minimum(u, u_max), u_min)

        x = md.rk4(x, u)

        states = np.hstack((states, x))
        controls = np.hstack((controls, u))
        t += md.DT

    md.visualize(states, controls=controls, ref_state=x_ref)
    md.animate(states)
