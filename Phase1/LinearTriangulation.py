import numpy as np

def skew(x):
     return np.array([[ 0  , -x[2],  x[1]],
                     [x[2],    0 , -x[0]],
                     [-x[1], x[0],    0]])
def homogenize(x):
    """
    N x m -> N x (m+1)
    """
    X = np.concatenate((x, np.ones((x.shape[0],1))),axis=1)
    return X

def create_projection_matrix(K, C, R):
    T = R @ C
    print(T.shape, R.shape, C.shape)
    P = K @ np.hstack((R, T))
    return P

def LinearTriangulation(K, C_ref, R_ref, C, R, V):
    
    c1 = np.reshape(C_ref,(3,1))
    c2 = np.reshape(C,(3,1))
    r1 = np.array(R_ref)
    r2 = np.array(R)
    v1 = V[:, 0, :]
    v2 = V[:, 1, :]
    v1 = homogenize(v1)
    v2 = homogenize(v2)
    P1 = create_projection_matrix(K, c1, r1)
    P2 = create_projection_matrix(K, c2, r2)

    X = []
    for p1,p2 in zip(v1,v2):
        a1 = skew(p1) @ P1 
        a2 = skew(p2) @ P2
        A = np.vstack((a1,a2))

        _, D, VT = np.linalg.svd(A)
        x = VT[np.argmin(D), :]
        x = x/x[3]
        x = x[0:3]
        x = np.array(x)
        X.append(x)
    X = np.vstack(X)
    print(X)
    # print(X)

    return X