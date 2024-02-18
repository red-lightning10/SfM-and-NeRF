import numpy as np
import cv2

def correction(C,R):
    if np.linalg.det(R)<0:
        return -C,-R
    else:
        return C,R
def get_cam_pose(E):
    U,D,V = np.linalg.svd(E)
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    u = U[:,2]
    C = [u,-u,u,-u]
    r_1 = np.dot(U,np.dot(W,V))
    r_2 = np.dot(U,np.dot(W.T,V))
    R = [r_1,r_1,r_2,r_2]
    for i in range(4):
       C[i],R[i] = correction(C[i],R[i])
    return C,R
