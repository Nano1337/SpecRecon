from weak_detect import *
import os
import random
import cv2

def load_data():
    '''Objective: Load Train Data Image into Single Folder'''
    # Load images from 22 patient folders of train set
    list = [i for i in range(0, 1220)]
    random.shuffle(list)
    counter = 0
    base_path = "D:/GLENDA_v1.5_no_pathology/no_pathology/testframes"
    target_path = "D:/weak_data/weak_test_images"
    for folder in os.listdir(base_path):
        folder_path = base_path + "/" + folder
        for file in os.listdir(folder_path):
            # Move images into one folder, randomize numerical assignment, convert to .tif
            file_path = folder_path + "/" + file
            read = cv2.imread(file_path)
            outfile = "/" + str(list[counter]) + '.tif'
            cv2.imwrite(target_path + outfile, read)
            print("Moving file " + str(counter))
            counter += 1

def load_masks():
    '''Run weak label algo through each image to store corresponding binary mask'''
    base_path = "D:/weak_data/weak_train_images/"
    target_path = "D:/weak_data/weak_train_masks/"
    for i in range(2230, 12218):
        file_path = base_path + str(i) + ".tif"
        read = cv2.imread(file_path)
        _, mask = run_weak_detect(read)
        outfile = "mask_" + str(i) + ".tif"
        cv2.imwrite(target_path + outfile, mask)
        print("Making mask " + str(i))

