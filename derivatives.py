import numpy as np
import modeling as md
from jax import jacfwd, jit
import numpy as np
import time 

l1 = l2 = 0.5
m1 = m2 = 1
g = 9.81

def M(x):
    M = np.array(
        [
            [
                l1 * l1 * m1 + l2 * l2 * m2 + l1 * l1 * m2 + 2 * l1 * l2 * m2 * np.cos(x[1, 0]),
                l2 * l2 * m2 + l1 * l2 * m2 * np.cos(x[1, 0]),
            ],
            [l2 * l2 * m2 + l1 * l2 * m2 * np.cos(x[1, 0]), l2 * l2 * m2],
        ]
    )
    return M

def C(x):
    C = np.array(
        [
            [
                -2 * x[3, 0] * l1 * m2 * l2 * np.sin(x[1, 0]),
                -x[3, 0] * l1 * m2 * l2 * np.sin(x[1, 0]),
            ],
            [x[2, 0] * l1 * m2 * l2 * np.sin(x[1, 0]), 0],
        ]
    )
    return C

def G(x):
    G = np.array(
        [
            [
                -g * m1 * l1 * np.sin(x[0, 0])
                - g * m2 * (l1 * np.sin(x[0, 0]) + l2 * np.sin(x[0, 0] + x[1, 0]))
            ],
            [-g * m2 * l2 * np.sin(x[0, 0] + x[1, 0])],
        ]
    )
    return G

def dM_dx(x):
    jac = np.zeros((2,2,4))
    q1 = x[0,0]
    q2 = x[1,0]

    dM11_dq2 = 2 * l1 * l2 * m2 * np.sin(q2)
    dM12_dq2 = l1 * l2 * m2 * np.sin(q2)
    dM21_dq2 = l1 * l2 * m2 * np.sin(q2)
    dM22_dq2 = 0
    jac[:,:,1] = np.array([[dM11_dq2, dM12_dq2], [dM21_dq2, dM22_dq2]])

    return jac

def dC_dx(x):
    q1 = x[0,0]
    q2 = x[1,0]
    q1_dot = x[2,0]
    q2_dot = x[3,0]

    jac = np.zeros((2,2,4))

    jac[:,:,1] = np.array([
        [-2 * q2_dot * l1 * m2 * l2 * np.cos(q2), -q2_dot * l1 * m2 * l2 * np.cos(q2)],
        [q1_dot * l1 * m2 * l2 * np.cos(q2), 0]
    ])

    jac[:,:,2] = np.array([
        [0, 0],
        [l1 * m2 * l2 * np.sin(q2), 0]
    ])

    jac[:,:,3] = np.array([
        [-2 * l1 * m2 * l2 * np.sin(q2), -l1 * m2 * l2 * np.sin(q2)],
        [0, 0]
    ])


    return jac
    
def dG_dx(x):
    q1 = x[0,0]
    q2 = x[1,0]
    q1_dot = x[2,0]
    q2_dot = x[3,0]
    jac = np.zeros((2,4))
    
    jac[0,0] =  -(m1 * g * l1 * np.cos(q1) + m2 * g * (l1 * np.cos(q1) + l2 * np.cos(q1 + q2)))
    jac[0,1] = -m2 * g * l2 * np.cos(q1 + q2)
    jac[1,0] = -m2*g*l2*np.cos(q1+q2)
    jac[1,1] = -m2*g*l2*np.cos(q1+q2)

    return jac
    


# def dynamics_dx(x, u):
#     jac = np.zeros((4,4))
#     M_inv = np.linalg.inv(M(x))
#     q_dot = x[2:,:]

#     # print("M_inv", M_inv.shape)
#     # print("dM_dx", dM_dx(x).shape)

#     tmp11 = np.tensordot(M_inv, dM_dx(x), ([0],[0]))
#     # print("M_inv@dM_dx", tmp11.shape)
#     tmp12 = np.tensordot(tmp11, M_inv, ([0],[0]))
#     # print("M_inv@dM_dx@M_inv", tmp12.shape)
#     tmp13 = u - C(x)@q_dot + G(x)

#     term1 = (tmp12@tmp13).reshape(2,4)
#     # print("term1", term1.shape)
#     # print(term1)


    
#     dqdot_dx = np.zeros((2,4))
#     dqdot_dx[:,2:] = np.eye(2)

#     # print("dC_dx", dC_dx(x).shape)
#     # print("q_dot", q_dot.shape)

#     tmp21 = np.tensordot(dC_dx(x), q_dot, ([1], [0])).reshape(2,4)
#     # print("tmp21", tmp21.shape)
#     tmp22 = C(x)@dqdot_dx
#     # print("tmp22", tmp22.shape)
#     tmp23 = tmp21 + tmp22
#     # print("tmp23",tmp23.shape)

#     # print("dG_dx", dG_dx(x).shape)

#     term2 = M_inv @ (-tmp23 + dG_dx(x))
#     # print(term2)

#     # Combine terms
#     dqddot_dx = term1 + term2  # Shape: (2, 4)

#     jac = np.block([
#         [dqdot_dx],  # Derivative of \dot{q}
#         [dqddot_dx]  # Derivative of \ddot{q}
#     ])

#     return jac


def dynamics_dx(x, u):
    jac = np.zeros((4,4))
    M_inv = np.linalg.inv(M(x))
    q_dot = x[2:,:]
    dqdot_dx = np.zeros((2,4))
    dqdot_dx[:,2:] = np.eye(2)


    tmp11 = np.tensordot(M_inv, dM_dx(x), ([0],[0]))
    tmp12 = np.tensordot(tmp11, M_inv, ([0],[0]))
    tmp13 = u - C(x)@q_dot + G(x)

    term1 = (tmp12@tmp13).reshape(2,4)


    tmp21 = np.tensordot(dC_dx(x), q_dot, ([1], [0])).reshape(2,4)
    tmp22 = C(x)@dqdot_dx
    tmp23 = tmp21 + tmp22

    term2 = M_inv @ (-tmp23 + dG_dx(x))

    dqddot_dx = term1 + term2

    jac = np.block([
        [dqdot_dx], 
        [dqddot_dx]
    ])

    return jac

def dynamics_du(x, u):
    jac = np.zeros((4,2))
    M_inv = np.linalg.inv(M(x))

    jac[2:,:] = M_inv

    return jac


# RK4 Jacobian dx
def rk4_dx(x, u, dt = 0.05):
    ### TO-DO: Write Jacobian df/dx
    k1 = md.dynamics(x, u)
    k2 = md.dynamics(x + 0.5 * dt * k1, u)
    k3 = md.dynamics(x + 0.5 * dt * k2, u)
    
    f1 = dynamics_dx(x, u)
    f2 = dynamics_dx(x + 0.5 * dt * k1, u)@(np.eye(4)+0.5*dt*f1)
    f3 = dynamics_dx(x + 0.5 * dt * k2, u)@(np.eye(4)+0.5*dt*f2)
    f4 = dynamics_dx(x + dt * k3, u)@(np.eye(4)+dt*f3)
    jac = np.eye(4) + (dt/6.)*(f1 + 2. * f2 + 2. * f3 + f4)
    return jac
    ### END of TO_DO

# RK4 Jacobian du
def rk4_du(x, u, dt = 0.05):
    ### TO-DO: Write Jacobian df/du
    k1 = md.dynamics(x, u)
    k2 = md.dynamics(x + 0.5 * dt * k1, u)
    k3 = md.dynamics(x + 0.5 * dt * k2, u)

    f1 = dynamics_du(x, u)
    f2 = dynamics_du(x + 0.5 * dt * k1, u) + dynamics_dx(x+0.5*dt*k1, u)@(0.5*dt*f1)
    f3 = dynamics_du(x + 0.5 * dt * k2, u) + dynamics_dx(x+0.5*dt*k2, u)@(0.5*dt*f2)
    f4 = dynamics_du(x + dt * k3, u) + dynamics_dx(x+dt*k3, u)@(dt*f3)
    jac =(dt/6.)*(f1 + 2. * f2 + 2. * f3 + f4)
    return jac
    ### END of TO-DO


def errors(x_bar, u_bar):    
    dx_auto = jit(jacfwd(md.dynamics, 0))(x_bar, u_bar).reshape(4,4)
    dx_exact =  dynamics_dx(x_bar, u_bar)

    print("dynamics_dx error:", np.linalg.norm(dx_auto- dx_exact))
    # print(dx_auto)
    # print(dx_exact)


    du_auto = jit(jacfwd(md.dynamics, 1))(x_bar, u_bar).reshape(4,2)
    du_exact =  dynamics_du(x_bar, u_bar)

    print("dynamics_du error:", np.linalg.norm(du_auto- du_exact))
    # print(du_auto)
    # print(du_exact)

    t1 = time.time()
    A = jit(jacfwd(md.rk4, 0))(x_bar, u_bar).reshape(4, 4)
    B = jit(jacfwd(md.rk4, 1))(x_bar, u_bar).reshape(4, 2)
    t2 = time.time()
    A_ = rk4_dx(x_bar, u_bar)
    B_ = rk4_du(x_bar, u_bar)
    t3 = time.time()

    print("rk4_dx error:", np.linalg.norm(A-A_))
    print("rk4_du error:", np.linalg.norm(B-B_))
    # print(t2-t1, t3-t2)




if __name__ == "__main__":

    x_bar = np.array([[np.pi,0.5,0.8,0.9]]).T
    u_bar = np.array([[0.6,0.9]]).T


    for k in range(10):
        x_bar = np.random.rand(4,1)*np.pi
        u_bar = np.random.rand(2,1)*20 - 10

        errors(x_bar, u_bar)
