import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import jax.numpy as jnp
import scienceplots
import time

plt.style.use(["science", "no-latex", "nature", "grid"])
plt.rcParams.update({"figure.dpi": "200"})


# x = [q1,q2,q1_d,q2_d]
# u = [u1,u2]

l1 = l2 = 0.5
m1 = m2 = 1
g = 9.81
DT = 0.01


def dynamics(x, u):

    M = jnp.array(
        [
            [
                l1 * l1 * m1 + l2 * l2 * m2 + l1 * l1 * m2 + 2 * l1 * l2 * m2 * jnp.cos(x[1, 0]),
                l2 * l2 * m2 + l1 * l2 * m2 * jnp.cos(x[1, 0]),
            ],
            [l2 * l2 * m2 + l1 * l2 * m2 * jnp.cos(x[1, 0]), l2 * l2 * m2],
        ]
    )
    C = jnp.array(
        [
            [
                -2 * x[3, 0] * l1 * m2 * l2 * jnp.sin(x[1, 0]),
                -x[3, 0] * l1 * m2 * l2 * jnp.sin(x[1, 0]),
            ],
            [x[2, 0] * l1 * m2 * l2 * jnp.sin(x[1, 0]), 0],
        ]
    )
    G = jnp.array(
        [
            [
                -g * m1 * l1 * jnp.sin(x[0, 0])
                - g * m2 * (l1 * jnp.sin(x[0, 0]) + l2 * jnp.sin(x[0, 0] + x[1, 0]))
            ],
            [-g * m2 * l2 * jnp.sin(x[0, 0] + x[1, 0])],
        ]
    )

    x_dot = jnp.vstack((x[2:, :], jnp.linalg.inv(M) @ (u - C @ x[2:, :] + G)))

    return x_dot


def rk4(xk, uk, dt=DT):
    f1 = dynamics(xk, uk)
    f2 = dynamics(xk + f1 * dt / 2, uk)
    f3 = dynamics(xk + f2 * dt / 2, uk)
    f4 = dynamics(xk + f3 * dt, uk)

    return xk + (dt / 6) * (f1 + 2 * f2 + 2 * f3 + f4)


def visualize(states):

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(states[0, :])
    ax2.plot(states[1, :])

    plt.show(block=False)


def animate(states):
    fig, ax = plt.subplots()

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    (line,) = ax.plot([], [], "o-", lw=1)

    def update(frame):

        x1 = l1 * np.sin(states[0, frame])
        y1 = -l1 * np.cos(states[0, frame])
        x2 = x1 + l2 * np.sin(states[0, frame] + states[1, frame])
        y2 = y1 - l2 * np.cos(states[0, frame] + states[1, frame])

        line.set_data([0, x1, x2], [0, y1, y2])

        return (line,)

    time_steps = states.shape[1]

    plt.gca().set_aspect("equal", adjustable="box")

    ani = FuncAnimation(fig, update, frames=time_steps, blit=True, interval=10)

    plt.show()


if __name__ == "__main__":

    # scenario 1
    initial_state = np.array([[0, 0, 0, 0]]).T
    u = np.array([[0, 0]]).T

    # scenario 2
    initial_state = np.array([[np.pi / 2, 0, 0, 0]]).T
    u = np.array([[l1 * m1 * g + (l1 + l2) * m2 * g, l2 * m2 * g]]).T

    # scenario 3
    initial_state = np.array([[np.pi / 2, -np.pi / 2, 0, 0]]).T
    u = np.array([[l1 * m1 * g + l1 * m2 * g, 0]]).T

    # scenario 4
    initial_state = np.array([[np.pi, 0, 0, 0]]).T
    u = np.array([[0, 0]]).T

    # simulation
    T = 5
    t = 0

    states = initial_state
    x = np.copy(initial_state)

    t1 = time.time()

    while t < T:

        x = rk4(x, u)

        states = np.hstack((states, x))

        t += DT

    print(time.time() - t1)
    print(states.T)

    visualize(states)

    animate(states)
