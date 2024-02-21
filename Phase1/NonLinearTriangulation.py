import numpy as np
from scipy.optimize import least_squares
from LinearTriangulation import homogenize, create_projection_matrix

def NonLinearTriangulation(K, C_ref, R_ref, C, R, X, V):
    
    P1 = create_projection_matrix(K, C_ref, R_ref)
    P2 = create_projection_matrix(K, C, R)
    v1 = homogenize(V[:, 0, :])
    v2 = homogenize(V[:, 1, :])
    X_nlt = []
    print(X.shape, v1.shape, v2.shape)
    for p1, p2, Xi in zip(v1, v2, X):
        # print(Xi)
        output = least_squares(fun=least_squares_fn, x0=Xi, args=(P1, P2, p1, p2))
        Xi = output.x
        X_nlt.append(Xi)

    return np.array(X_nlt)

def least_squares_fn(X, P1, P2, pt1, pt2):

    X_h = np.hstack((X, 1))

    u1, v1, u2, v2 = pt1[0], pt1[1], pt2[0], pt2[1]
    print(u1, P1[0, :] @ X_h, P1[2, :] @ X_h)

    error1 = np.array([u1 - (P1[0, :] @ X_h) / (P1[2, :] @ X_h), 
                      v1 - (P1[1, :] @ X_h) / (P1[2, :] @ X_h)])
    error2 = np.array([u2 - (P2[0, :] @ X_h) / (P2[2, :] @ X_h),
                      v2 - (P2[1, :] @ X_h) / (P2[2, :] @ X_h)])

    # return np.concatenate((error1**2, error2**2))
    # print(error1**2 + error2**2)
    return np.sum(error1**2 + error2**2)
