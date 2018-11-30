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

featureArray = np.zeros((8000, 3780), np.float32)
labelArray = np.zeros((8000, 1), np.int32)
svm_trainer = cv.ml.SVM_create()
svm = cv.ml_SVM.getS
svm_trainer.setType(cv.ml.SVM_C_SVC)
svm_trainer.setKernel(cv.ml.SVM_LINEAR)
svm_trainer.setC(0.01)

# train
svm_trainer.train(featureArray, cv.ml.ROW_SAMPLE, labelArray)
