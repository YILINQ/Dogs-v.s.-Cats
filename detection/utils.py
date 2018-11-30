import cv2 as cv
import numpy as np
import os
from skimage.feature import hog
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
from hog_and_svm import extract_hog, svm

def process_data(data_path, labels, flag):
    """
    Construct a hog class and use it to extract features
    Then put the features into a .pickle file for future training purpose
    variable flag could be "positive" or "negative" to identify the tyoe of dataset
    """
    files = os.listdir(data_path)
    data = []
    label = []
    for image in files:
        hog = extract_hog(64, 12, 8, 2, True, True)
        path = os.path.join(data_path, image)
        img = cv.imread(path)
        features = hog.construct_feature(img)
        data.append(list(features))
        label.append(list(labels))
    output = [data, label]
    with open("TrainData" + str(flag) + ".pickle", 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
