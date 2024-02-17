import cv2
import numpy as np
import os
import sys
import glob
from matplotlib import pyplot as plt
from EssentialMatrixFromFundamentalMatrix import EfromF
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
    # Read feature descriptors
    feature_descriptors = []
    #create nested dictionary for feature matches
    feature_matches = create_feature_match_dict(len(image_paths))

    for i in range(len(descriptor_files)):
        feature_matches = ReadFeatureDescriptors(descriptor_files[i], feature_matches, i)
        for j in range(i + 1, len(image_paths)):
            result_img = plot_feature_correspondences(images[i], images[j], feature_matches[i + 1][j + 1])
            cv2.imwrite(os.path.join(results_path, 'correspondences_before_RANSAC' + str(i+1) + '_' + str(j+1) + '.png'), result_img)

            #RANSAC filtering
            filtered_matches = GetInlierRANSAC(feature_matches[i + 1][j + 1], 5e-3, 1000)

            RANSAC_result_img = plot_feature_correspondences(images[i], images[j], filtered_matches)
            cv2.imwrite(os.path.join(results_path, 'correspondences_RANSAC' + str(i+1) + '_' + str(j+1) + '.png'), RANSAC_result_img)
            F = EstimateFundamentalMatrix(filtered_matches)
            E = EfromF(F,K)
            print(np.linalg.matrix_rank(F))
            print(E)

if __name__ == "__main__":
    main()
