def DisambiguateCameraPose(Cset, Rset, Xset):

    max_points_in_front = 0
    for i in range(len(Cset)):
        r3 = Rset[i][2]
        C = Cset[i]

        num_points_in_front = [r3*(Xset[i] - C) > 0 for i in range(len(Xset))]

        if sum(num_points_in_front) > max_points_in_front:
            max_points_in_front = sum(num_points_in_front)
            C = Cset[i]
            R = Rset[i]

    return C, R
