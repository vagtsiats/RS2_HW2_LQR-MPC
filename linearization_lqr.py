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


if __name__ == "__main__":

    x_bar = np.array([[np.pi, 0.0, 0.0, 0.0]]).T
    u_bar = np.array([[0.0, 0.0]]).T

    print(linearize(x_bar, u_bar)[0])
