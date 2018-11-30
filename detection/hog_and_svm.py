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

class extract_hog:
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


class svm:
    """
    This class takes in both positive and negative datasets to train svm
    We read the data from pickle file

    """
    def __init__(self, pos, neg):
        self.pos_path = pos
        self.neg_path = neg
        self.extract_hog = extract_hog(box=64, orientations=64, pixel_per_cell=12, cells_per_block=8, vis=True, transform_sqrt=True)
        self.svc = None
        self.scale = None

    def train(self):
        positive, negative = [[], []],  [[], []]
        for path in self.pos_path:
            with open(path, 'rb') as handle:
                sample = pickle.load(handle)
                positive[0] += sample[0]
                positive[1] += sample[1]

        for path in self.neg_path:
            with open(path, 'rb') as handle:
                sample = pickle.load(handle)
                negative[0] += sample[0]
                negative[1] += sample[1]
        positive_train_data, negative_train_data = np.asarray(positive), np.asarray(negative)
        positive_train_label, negative_train_label = positive[1], negative[1]
        s = StandardScaler().fit(np.vstack((positive_train_data, negative_train_data).astype(np.float64)))
        X = s.transform(np.vstack((positive_train_data, negative_train_data).astype(np.float64)))
        Y = np.asarray(positive_train_label + negative_train_label)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

        svc = SVC(gamma="scale")
        svc.fit(x_train, y_train)
        # acc = svc.score(x_test, y_test)
        self.svc = svc
        self.scale = s

    def classify(self):
        return self.svc.predict(self.scale.transform([self.extract_hog]))

if __name__ == "__main__":
    pass
