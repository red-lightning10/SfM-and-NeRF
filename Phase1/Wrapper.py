import cv2
import numpy as np
import os
import sys
import glob
from matplotlib import pyplot as plt

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
        # cv2.line(concatenated_image, (source_x, source_y), (target_x, target_y), (0, 255, 0), 1)
    
    return concatenated_image
# def EstimateFundamentalMatrix():

# def GetInlierRANSAC():


def main():
    
    path = os.getcwd()
    calibration_file = os.path.join(os.path.join(path, 'Data'), 'calibration.txt')
    # print(calibration_file)
    K = ReadCalibrationFile(calibration_file)

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
    print(np.shape(images))

    print(image_paths)
    print(descriptor_files)
    # Read feature descriptors
    feature_descriptors = []
    #create nested dictionary for feature matches
    feature_matches = create_feature_match_dict(len(image_paths))

    print(feature_matches)
    for i in range(len(descriptor_files)):
        feature_matches = ReadFeatureDescriptors(descriptor_files[i], feature_matches, i)
        print(feature_matches[i + 1].keys())
        for j in range(i + 1, len(image_paths)):
            result_img = plot_feature_correspondences(images[i], images[j], feature_matches[i + 1][j + 1])
            print(i+1, j+1)
            cv2.imwrite(os.path.join(results_path, 'correspondences_before_RANSAC' + str(i+1) + '_' + str(j+1) + '.png'), result_img)

if __name__ == "__main__":
    main()
