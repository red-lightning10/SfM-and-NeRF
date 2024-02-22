import numpy as np
from LinearTriangulation import homogenize

def LinearPnP(x_2d, X_3d, K):

    X, Y, Z = X_3d.T
    x_2d_h = homogenize(x_2d)
    K_inv = np.linalg.inv(K)
    print(x_2d_h[0, 1])

    # x = [K_inv @ x_2d_h[i].T for i in range(len(x_2d_h))]
    # print(x)
    print(x_2d_h.shape)
    A = []
    for i in range(len(X_3d)):
        
        if not i:
            A = [X[i], Y[i], Z[i], 1, 0, 0, 0, 0, -x_2d_h[i, 0]*X[i], -x_2d_h[i, 0]*Y[i], -x_2d_h[i, 0]*Z[i], -x_2d_h[i, 0]]
        else:
            A = np.vstack((A, [X[i], Y[i], Z[i], 1, 0, 0, 0, 0, -x_2d_h[i, 0]*X[i], -x_2d_h[i, 0]*Y[i], -x_2d_h[i, 0]*Z[i], -x_2d_h[i, 0]]))
        
        A = np.vstack((A, [0, 0, 0, 0, X[i], Y[i], Z[i], 1, -x_2d_h[i, 1]*X[i], -x_2d_h[i, 1]*Y[i], -x_2d_h[i, 1]*Z[i], -x_2d_h[i, 1]]))
    
    print(A.shape)

    _, _, VT = np.linalg.svd(A)
    P = VT[-1, :].reshape(3, 4)
    R = K_inv @ P[:, 0:3]

    U, S, VT_ = np.linalg.svd(R)
    R = U @ VT_
    print(S)
    gamma = S[0]

    T = K_inv @ P[:, 3]/gamma

    if np.linalg.det(R) < 0:
        R = -R
        T = -T

    C = -R.T @ T
    return R, C

