import cv2 as cv
import numpy as np
import os
from skimage.feature import hog
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
import pickle

class data_processor:
    def __init__(self, posPath, negPath, hog):
        self.posDir = posPath
        self.negDir = negPath
        self.pos_array = []
        self.neg_array = []
        self.pos_label = []
        self.neg_label = []
        self.hog_extractor = hog
    def load_data(self):
        posFiles = os.listdir(self.posDir)
        for image in posFiles:
            # load positive data
            fileName = os.path.join(self.posDir, image)
            # print(fileName)
            img = cv.imread(fileName)
            img = cv.resize(img, (64, 128), interpolation=cv.INTER_AREA)
            # compute 3780 dimension vector hist
            hist = self.hog_extractor.compute(img, (8, 8))
            # can write to a pickle file
            self.pos_array.append(hist.reshape(-1, 15876)[0])
            self.pos_label.append(1)

        negFiles = os.listdir(self.negDir)
        for image in negFiles:
                # load positive data
            fileName = os.path.join(self.negDir, image)
            # print(fileName)
            img = cv.imread(fileName)
            img = cv.resize(img, (64, 128), interpolation=cv.INTER_AREA)
            # compute 3780 dimension vector hist
            hist = self.hog_extractor.compute(img, (8, 8))
            # can write to a pickle file
            self.neg_array.append(hist.reshape(-1, 15876)[0])
            self.neg_label.append(0)

    def transform_data(self):
        pos_data = np.asarray(self.pos_array)
        neg_data = np.asarray(self.neg_array)
        unscale = np.vstack((pos_data, neg_data)).astype(np.float64)
        return pos_data, neg_data, self.pos_label, self.neg_label, unscale


class svm:
    def __init__(self, pos_data, neg_data, pos_label, neg_label, unscale):
        self.pos_data = pos_data
        self.neg_data = neg_data
        self.unscale = unscale
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.scaler = StandardScaler().fit(self.unscale)
        self.X = self.scaler.transform(unscale)
        self.Y = np.asarray(self.pos_label + self.neg_label)
        self.svc = SVC(gamma='scale')

    def train_svm(self):
        train_data, test_data, train_label, test_label = train_test_split(self.X, self.Y, test_size=0.2,
                                                                          random_state=0)
        self.svc.fit(train_data, train_label)
        # score = self.svc.score(test_data, test_label)
        # print("Accuracy:" + str(score * 100.0) + "%")


class detect:
    def __init__(self, svm, image_path, hog, grouped):
        self.svm = svm
        self.image_path = image_path
        self.hog_extractor = hog
        self.draw_recs = []
        self.grouped = grouped

    def detect_window(self):
        posFiles = os.listdir(self.image_path)
        test = []
        for image in posFiles:
            fileName = os.path.join(self.image_path, image)
            img = cv.imread(fileName)
            winSize = (int(img.shape[0] * 0.45), int(img.shape[1] * 0.45))
            stride = (int(winSize[0] * 0.2), int(winSize[1] * 0.2))
            for x_ in range(winSize[0], img.shape[1], stride[0]):
                for y_ in range(winSize[1], img.shape[0], stride[1]):
                    x = (x_ - winSize[0], x_)
                    y = (y_ - winSize[1], y_)
                    a = img[y[0]:y[1], x[0]:x[1], :]
                    img_ = cv.resize(a, (64, 128), interpolation=cv.INTER_AREA)
                    new_hist = self.hog_extractor.compute(img_, (8, 8))
                    test = [new_hist.reshape(-1, 15876)[0]]
                    if self.svm.svc.predict(test) == [1]:
                        self.draw_recs.append((x[0], y[0], x[1], y[1]))
            self.draw(img)

    def draw(self, img):
        grouped = self.grouped
        if grouped:
            x_begins, y_begins, x_ends, y_ends = [], [], [], []
            # print(recs)
            for rec in self.draw_recs:
                x_begins.append(rec[0])
                y_begins.append(rec[1])
                x_ends.append(rec[2])
                y_ends.append(rec[3])
            if x_begins and y_begins and x_ends and y_ends:
                x = min(x_begins)
                y = min(y_begins)
                x_ = max(x_ends)
                y_ = max(y_ends)
                cv.rectangle(img, (x, y), (x_, y_), (255, 255, 0), thickness=5)
        else:
            for rec in self.draw_recs:
                x = rec[0]
                x_ = rec[2]
                y = rec[1]
                y_ = rec[3]
                cv.rectangle(img, (x, y), (x_, y_), (0, 0, 255), thickness=5)
        import random
        cv.imwrite("detection" + str(random.randint(1, 10000)) + ".jpg", img)

if __name__ == "__main__":
    hog_extractor = cv.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)

    DP_dog = data_processor(posPath="dog_pos", negPath="dog_neg", hog=hog_extractor)
    DP_dog.load_data()
    pos_data, neg_data, pos_label, neg_label, unscale = DP_dog.transform_data()
    svm_trainer_dog = svm(pos_data, neg_data, pos_label, neg_label, unscale)
    svm_trainer_dog.train_svm()
    dog_detecor = detect(svm_trainer_dog, image_path="test_svm_dog", hog=hog_extractor, grouped=False)
    dog_detecor.detect_window()
    dog_detecor = detect(svm_trainer_dog, image_path="test_svm_dog", hog=hog_extractor, grouped=True)
    dog_detecor.detect_window()

    # DP_cat = data_processor(posPath="cat_pos", negPath="cat_neg", hog=hog_extractor)
    # DP_cat.load_data()
    # pos_data, neg_data, pos_label, neg_label, unscale = DP_cat.transform_data()
    # svm_trainer_cat = svm(pos_data, neg_data, pos_label, neg_label, unscale)
    # svm_trainer_cat.train_svm()
    # cat_detecor = detect(svm_trainer_cat, image_path="test_svm_cat", hog=hog_extractor, grouped=False)
    # cat_detecor.detect_window()
    # cat_detecor = detect(svm_trainer_cat, image_path="test_svm_cat", hog=hog_extractor, grouped=True)
    # cat_detecor.detect_window()
