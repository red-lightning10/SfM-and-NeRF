import numpy as np

def skew(x):
     return np.array([[ 0  , -x[2],  x[1]],
                     [x[2],    0 , -x[0]],
                     [-x[1], x[0],    0]])
def triangulation(K,C,R,v1,v2):
    c1 = C[0]
    c2 = C[1]
    r1 = R[0]
    r2 = R[1]
    T1 = r1*c1
    T2 = r2*c2
    P1 = np.dot(K,np.dot(r1,np.hstack(np.identity(3),-c1)))
    P2 = np.dot(K,np.dot(r2,np.hstack(np.identity(3),-c2)))

    for p1,p2 in zip(v1,v2):
        a1 = skew(p1)*P1
        a2 = skew(p2)*P2
        A = np.vstack(a1,a2)

        U,D,V = np.linalg.svd(A)
        

