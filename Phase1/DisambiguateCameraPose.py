import numpy as np

def DisambiguateCameraPose(Cset, Rset, Xset):

    max_points_in_front = 0
    max_X_in_front = []
    for i in range(len(Cset)):
        r3 = Rset[i][:, 2]
        C = Cset[i]
        X = Xset[i]
        # print(X)
        num_points_in_front = []
        # num_points_in_front = [np.dot(r3,(X[j] - C)) > 0  for j in range(len(X))]
        num_points_in_front = [r3 @ (X[j] - C).T > 0 for j in range(len(X)) if X[j][2] > 0]
        if sum(num_points_in_front) > max_points_in_front:
            max_points_in_front = sum(num_points_in_front)
            C = Cset[i]
            R = Rset[i]
            max_X_in_front = Xset[i]
    print("max", np.shape(max_X_in_front), Xset[0].shape)
    return C.reshape(3,1), R, max_X_in_front
