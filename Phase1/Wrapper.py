import cv2
import numpy as np
import os
import sys
import glob
from matplotlib import pyplot as plt
from EssentialMatrixFromFundamentalMatrix import EfromF
from ExtractCameraPose import get_cam_pose
from LinearTriangulation import LinearTriangulation
from DisambiguateCameraPose import DisambiguateCameraPose
from NonLinearTriangulation import NonLinearTriangulation
from PnPRANSAC import PnPRANSAC
from NonLinearPnP import NonLinearPnP

def create_feature_match_dict(n):
    feature_matches = {}
    for i in range(1, n):
        feature_matches[i] = {}
        for j in range(i + 1, n + 1):
            feature_matches[i][j] = {}

    return feature_matches

def ReadCalibrationFile(calibration_file):
    with open(calibration_file, 'r') as f:
        lines = f.readlines()
        K = np.zeros((3, 3))
        for i, line in enumerate(lines):
            if i < 3:
                K[i, :] = [float(x) for x in line.split()]
            else:
                break
        return K

def ReadFeatureDescriptors(descriptor_file, feature_matches, i):
    with open(descriptor_file, 'r') as f:
        lines = f.readlines()
        for j in range(len(lines)):
            lines[j] = lines[j].split()
        nFeatures = int(lines[0][1])

        for j in range(1, len(lines)):
            nFeatures_kp = int(lines[j][0])
            image1_id = i + 1
            image1_kp_rgb = [int(lines[j][1]), int(lines[j][2]), int(lines[j][3])]
            image1_kp = [float(lines[j][4]), float(lines[j][5])]
            
            for k in range(nFeatures_kp - 1):
                image2_id = int(lines[j][6 + 3*k])
                image2_kp= [float(lines[j][7 + 3*k]), float(lines[j][8 + 3*k])]
                current_matches = list(feature_matches.get(image1_id).get(image2_id))
                current_matches.append((image1_kp, image2_kp))
                feature_matches[image1_id][image2_id] = current_matches
                # print(image1_id, image2_id, image1_kp, image2_kp)
        # print(lines)
        # feature_descriptors = np.array(lines, dtype=np.float32)
    # print(feature_matches)
    return feature_matches

def plot_feature_correspondences(source, target, matches):

    concatenated_image = np.concatenate((source, target), axis = 1)

    for match in matches:
        source_x = int(match[0][0])
        source_y = int(match[0][1])
        target_x = int(match[1][0]) + source.shape[1]
        target_y = int(match[1][1])
        cv2.circle(concatenated_image, (source_x, source_y), 2, (0, 0, 255), -1)
        cv2.circle(concatenated_image, (target_x, target_y), 2, (0, 0, 255), -1)
        cv2.line(concatenated_image, (source_x, source_y), (target_x, target_y), (0, 255, 0), 1)
    
    return concatenated_image

def EstimateFundamentalMatrix(matches_array):

    A = []
    if len(matches_array) < 8 or len(matches_array) < 8: 
        print("Fundamental Matrix needs more sets of points. Aborting.")
        return A
    
    x = matches_array[:, 0, 0]
    y = matches_array[:, 0, 1]
    xm = matches_array[:, 1, 0]
    ym = matches_array[:, 1, 1]

    # choices = np.random.choice(np.arange(len(matches)), 8, replace=False)
    for i in range(len(x)):

        A.append([x[i]*xm[i], x[i]*ym[i], x[i], y[i]*xm[i], y[i]*ym[i], y[i], xm[i], ym[i], 1])

    _, _, V = np.linalg.svd(A)
    F = (np.transpose(V)[:, -1]).reshape(3, 3)

    U, S, V = np.linalg.svd(F)
    S = np.diag(S)
    S[2,2] = 0
    F = np.dot(U, np.dot(S, V))
    return F
    

def GetInlierRANSAC(matches, threshold, nIterations):

    A = []
    if len(matches) < 8 or len(matches) < 8: 
        print("Operation needs more sets of points. Aborting.")
        return A

    num_max_inliers = 0
    inliers = []
    for i in range(nIterations):
        choices = np.random.choice(np.arange(len(matches)), 8, replace=False)
        matches_array = np.array(matches)

        F = EstimateFundamentalMatrix(matches_array[choices])

        x2 = np.concatenate((matches_array[:, 1, :], np.ones((len(matches_array), 1))), axis=1)
        x1 = np.concatenate((matches_array[:, 0, :], np.ones((len(matches_array), 1))), axis=1)

        inliers_check = [np.abs(np.dot(x2[j], np.dot(F, np.transpose(x1[j])))) < threshold for j in range(len(matches))]
        num_inliers = np.sum(inliers_check)

        if num_inliers > num_max_inliers:

            inliers_indices = np.where(inliers_check)
            num_max_inliers = num_inliers

    inliers = matches_array[inliers_indices]
    return inliers

def get_epipole(F):
    U, _, V = np.linalg.svd(F)
    e1 = V[-1]
    e2 = U[:,-1]

    return e1/e1[-1], e2/e2[-1]

def get_epipolar_lines(F, pts1, pts2):

    lines_1 = [F.T @ np.array([pt[0], pt[1], 1]).T for pt in pts2]
    lines_2 = [F @ np.array([pt[0], pt[1], 1]).T for pt in pts1]

    return lines_1, lines_2

def plot_epipolar_lines(img1, img2, F, pts1, pts2):

    lines1, lines2 = get_epipolar_lines(F, pts1, pts2)
    i1 = img1.copy()
    i2 = img2.copy()
    for l1, l2 in zip(lines1, lines2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, x1 = 0, img1.shape[1]
        y0 = int(-l1[2]/l1[1])
        y1 = int((-l1[2] - l1[0]*x1)/l1[1])
        i1 = cv2.line(i1, (x0, y0), (x1, y1), color, 1)

        x0, x1 = 0, img2.shape[1]
        y0 = int(-l2[2]/l2[1])
        y1 = int((-l2[2] - l2[0]*x1)/l2[1])
        i2 = cv2.line(i2, (x0, y0), (x1, y1), color, 1)

    e1, e2 = get_epipole(F)
    e1 = int(e1[0]), int(e1[1])
    e2 = int(e2[0]), int(e2[1])
    #plot a point at the epipole
    i1 = cv2.circle(i1, e1, 3, (0, 0, 255), -1)
    i2 = cv2.circle(i2, e2, 3, (0, 0, 255), -1)

    return np.concatenate((i1, i2), axis = 1)

def main():
    
    path = os.getcwd()
    calibration_file = os.path.join(os.path.join(path, 'Data'), 'calibration.txt')
    # print(calibration_file)
    K = ReadCalibrationFile(calibration_file)
    print(K)
    results_path = os.path.join(path, "Results")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    #Read Descriptor files
    descriptor_files = []
    for i in os.listdir(os.path.join(path, 'Data')):
        if i.startswith('matching') and i.endswith('.txt'):
            descriptor_files.append(os.path.join(os.path.join(path, 'Data'), i))
    
    descriptor_files = sorted(descriptor_files)
    
    # Read images
    images = []
    image_paths = glob.glob(os.path.join(os.path.join(path, 'Data'), '*.png'))
    image_paths = sorted(image_paths)
    for i in range(len(image_paths)):
        img = cv2.imread(image_paths[i])
        images.append(img)

    #create nested dictionary for feature matches
    feature_matches = create_feature_match_dict(len(image_paths))
    filtered_matches = create_feature_match_dict(len(image_paths))
    for i in range(len(descriptor_files)):
        feature_matches = ReadFeatureDescriptors(descriptor_files[i], feature_matches, i)
        for j in range(i + 1, len(image_paths)):
            result_img = plot_feature_correspondences(images[i], images[j], feature_matches[i + 1][j + 1])
            cv2.imwrite(os.path.join(results_path, 'correspondences_before_RANSAC' + str(i+1) + '_' + str(j+1) + '.png'), result_img)

            #RANSAC filtering
            filtered_matches_list = GetInlierRANSAC(feature_matches[i + 1][j + 1], 5e-3, 1000)
            filtered_matches[i + 1][j + 1] = filtered_matches_list
            RANSAC_result_img = plot_feature_correspondences(images[i], images[j], filtered_matches_list)
            cv2.imwrite(os.path.join(results_path, 'correspondences_RANSAC' + str(i+1) + '_' + str(j+1) + '.png'), RANSAC_result_img)

    filtered_matches_array = np.array(filtered_matches.get(1).get(2))
    F = EstimateFundamentalMatrix(filtered_matches_array)
    E = EfromF(F,K)
    C,R = get_cam_pose(E)

    # plot_epipolar_result_img = plot_epipolar_lines(images[0], images[1], F, filtered_matches_array[:, 0, :], filtered_matches_array[:, 1, :])
    # cv2.imwrite(os.path.join(results_path, 'epipolar_lines' + str(1) + '_' + str(2) + '.png'), plot_epipolar_result_img)

    C0 = np.zeros((3,1))
    R0 = np.eye(3)

    X_lt = []
    print("fm", filtered_matches_array.shape)
    for Ci, Ri in zip(C,R):
        x_lt = LinearTriangulation(K, C0, R0, Ci, Ri, filtered_matches_array)
        print("x_lt", x_lt.shape)
        X_lt.append(x_lt)

    C, R, X = DisambiguateCameraPose(C, R, X_lt)
    X_nlt = NonLinearTriangulation(K, C0, R0, C, R, X, filtered_matches_array)
    print(X.shape)
    print(X_nlt.shape)

    
    # PnP
    x_2d = filtered_matches_array[:, 1, :]
    X_3d = X_nlt
    R, C = PnPRANSAC(X_3d, x_2d, K, threshold=20, nIterations=1000)
    
    R_new, C_new = NonLinearPnP(X_3d, x_2d, K, C, R)
    print(R_new, C_new)

if __name__ == "__main__":
    main()