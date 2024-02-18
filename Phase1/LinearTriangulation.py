import numpy as np

def skew(x):
     return np.array([[ 0  , -x[2],  x[1]],
                     [x[2],    0 , -x[0]],
                     [-x[1], x[0],    0]])
def homogenize(x):
    """
    N x m -> N x (m+1)
    """
    X = np.concatenate((x,np.ones((x.shape[0],1))),axis=1)
    return X

def triangulation(K,C,R,V):
    c1 = np.reshape(C[0],(3,1))
    c2 = np.reshape(C[1],(3,1))
    print(np.shape(c1))
    r1 = np.array(R[0])
    r2 = np.array(R[1])
    v1 = V[:, 0, :]
    v2 = V[:, 1, :]
    v1 = homogenize(v1)
    v2 = homogenize(v2)
    T1 = r1*c1
    T2 = r2*c2
    P1 = np.dot(K,np.dot(r1,np.hstack((np.identity(3),-c1))))
    P2 = np.dot(K,np.dot(r2,np.hstack((np.identity(3),-c2))))
    X = []
    for p1,p2 in zip(v1,v2):
        a1 = skew(p1)@P1 # @ is cross product
        a2 = skew(p2)@P2
        A = np.vstack((a1,a2))

        U,D,VT = np.linalg.svd(A)
        x = VT[np.argmin(D),:]
        x = x/x[3]
        x = x[0:3]
        x =np.array(x)
        X.append(x)
    X = np.vstack(X)
    return X
