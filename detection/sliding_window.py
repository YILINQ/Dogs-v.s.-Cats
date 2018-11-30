import cv2 as cv
import numpy as np
import os
from skimage.feature import hog
from hog_and_svm import extract_hog, svm
from utils import process_data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

pos_data, neg_data = None, None
class SlidingWindow:
    def __init__(self, image):
        self.hog_extractor = extract_hog(box=64, orientations=12, pixel_per_cell=8, cells_per_block=2, vis=True, transform_sqrt=True)
        self.svm = svm(pos=pos_data, neg=neg_data)

    def sliding_window(self):
        pass

if __name__ == "__main__":
    pass
