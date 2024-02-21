import numpy as np

def visibility_matrix(num_points,num_camera, C_set,R_set,X_set):
    V = np.zeros((num_points,num_camera))
    for i in range(num_camera):
        C = C_set[i]
        R = R_set[i]
        X = X_set[i]
        for j in range(num_points):
            r3 = R[:,2]
            C = C.reshape(-1,1)
            X = X.reshape(-1,1)
            if r3 @ (X[j] - C).T > 0:
                V[j,i] = 1
    return V
