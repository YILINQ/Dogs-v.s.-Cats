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

class HOG:
    def __init__(self, box, orientations, pixel_per_cell, cells_per_block, vis, transform_sqrt):
        self.box_size = box
        self.orientations = orientations
        self.pixel_per_cell = pixel_per_cell
        self.cells_per_block = cells_per_block
        self.visualize = vis
        self.transform_sqrt = transform_sqrt
        self.descriptor = None
        self.hog_image = None
    def get_hog_descriptor_and_image(self, image):
        return hog(image, orientations=self.orientations,
                                    pixels_per_cell=self.pixel_per_cell,
                                    cells_per_block=self.cells_per_block,
                                    visualize=self.visualize,
                                    transform_sqrt=self.transform_sqrt,
                                    feature_vector=False)
    def extract_hog(self, image):
        image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        h = cv.resize(image[:, :, 0], self.box_size)
        s = cv.resize(image[:, :, 1], self.box_size)
        v = cv.resize(image[:, :, 2], self.box_size)
        h_descriptor, h_image = self.get_hog_descriptor_and_image(h)
        s_descriptor, s_image = self.get_hog_descriptor_and_image(s)
        v_descriptor, v_image = self.get_hog_descriptor_and_image(v)
        return {"h_hog": h_descriptor,
                "s_hog": s_descriptor,
                "v_hog": v_descriptor,
                "h_img": h_image,
                "s_img": s_image,
                "v_img": v_image}

    def construct_feature(self, image):
        features = self.extract_hog(image)
        return np.hstack((features["h_hog"], features["s_hog"], features["v_hog"]))
