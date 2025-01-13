import cv2
import os
from os import listdir

# get the path or directory
folder_dir = "C:/Users/DELL/Downloads/NeRF_custom_sets-20240311T194517Z-001/NeRF_custom_sets/NeRF_custom_sets/test/"
for images in os.listdir(folder_dir):
    img = cv2.imread(folder_dir+images)
    img = cv2.resize(img, (200, 200), interpolation = cv2.INTER_AREA)
    cv2.imwrite(folder_dir+"r_"+images, img)


