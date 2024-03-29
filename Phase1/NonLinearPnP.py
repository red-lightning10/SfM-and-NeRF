import numpy as np
from scipy.optimize import least_squares

def quaternion_to_rotation_matrix(q):
    q = q / np.linalg.norm(q)
    R = np.zeros((3,3))
    R[0,0] = 1 - 2 * q[2]**2 - 2 * q[3]**2
    R[0,1] = 2 * q[1] * q[2] - 2 * q[0] * q[3]
    R[0,2] = 2 * q[1] * q[3] + 2 * q[0] * q[2]
    R[1,0] = 2 * q[1] * q[2] + 2 * q[0] * q[3]
    R[1,1] = 1 - 2 * q[1]**2 - 2 * q[3]**2
    R[1,2] = 2 * q[2] * q[3] - 2 * q[0] * q[1]
    R[2,0] = 2 * q[1] * q[3] - 2 * q[0] * q[2]
    R[2,1] = 2 * q[2] * q[3] + 2 * q[0] * q[1]
    R[2,2] = 1 - 2 * q[1]**2 - 2 * q[2]**2
    return R

def rotation_matrix_to_quaternion(R):
    q = np.zeros(4)
    R = np.array(R)
    print(R.shape)
    q[0] = 0.5 * np.sqrt(1 + R[0,0] + R[1,1] + R[2,2])
    q[1] = (R[2,1] - R[1,2]) / (4 * q[0])
    q[2] = (R[0,2] - R[2,0]) / (4 * q[0])
    q[3] = (R[1,0] - R[0,1]) / (4 * q[0])
    return q

def NonLinearPnP(X_3d, x_2d, K, C, R):

    # x_2d_h = np.hstack((x_2d, np.ones((x_2d.shape[0], 1))))
    # print(x_2d_h)
    q = rotation_matrix_to_quaternion(R)
    Xi = np.hstack((q, C))

    output = least_squares(fun=least_squares_fn_pnp, x0=Xi, args=(X_3d, x_2d, K))
    Xi = output.x
    q_new = Xi[0:4]
    C_new = Xi[4:]
    R_new = quaternion_to_rotation_matrix(q_new)
    
    return R_new, C_new


def least_squares_fn_pnp(params, X_3d, x_2d, K):
    q = params[0:4]
    C = params[4:]
    R = quaternion_to_rotation_matrix(q)
    error = []
    for i in range(len(X_3d)):
        x_2d_hat = K @ (R @ X_3d[i] + C)
        x_2d_hat = x_2d_hat / x_2d_hat[2]
        x_2d_hat = x_2d_hat[0:2]
        e = x_2d[i] - x_2d_hat
        error.append(e**2)

    return np.sum(error)