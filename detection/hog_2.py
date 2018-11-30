import cv2 as cv
import numpy as np
import os
from skimage.feature import hog
from sklearn import svm as sk_svm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle


class HOG:
    def __init__(self, windowSize, blockSize, blockStride, cellSize, nBins):
        self.windowSize = windowSize
        self.blockSize = blockSize
        self.blockStride = blockStride
        self.cellSize = cellSize
        self.nBins = nBins
        self.hog_extractor = cv.HOGDescriptor(self.windowSize,
                                              self.blockSize,
                                              self.blockStride,
                                              self.cellSize,
                                              self.nBins)
    def compute(self, src, size):
        return self.hog_extractor.compute(src, size)


class pre_processor:
    def __init__(self, featureNum, PosNum, NegNum, hog):
        self.featureNum = featureNum
        self.PosNum = PosNum
        self.NegNum = NegNum
        self.featureArray = np.zeros((PosNum+NegNum, featureNum), np.float32)
        self.labelArray = np.zeros((PosNum+NegNum, 1), np.int32)
        self.hog_extractor = hog

    def load_pos_and_neg_data(self, posPath, negPath):
        posFiles = os.listdir(posPath)
        pos_index = 0
        for image in posFiles:
            # load positive data
            fileName = os.path.join(posPath, image)
            # print(fileName)
            img = cv.imread(fileName)
            img = cv.resize(img, (64, 128), interpolation=cv.INTER_AREA)
            # compute 3780 dimension vector hist
            hist = self.hog_extractor.compute(img, (8, 8))
            # can write to a pickle file
            for j in range(self.featureNum):
                self.featureArray[pos_index, j] = hist[j]
            self.labelArray[pos_index, 0] = 1
            pos_index += 1
        negFiles = os.listdir(negPath)
        neg_index = 1
        for image in negFiles:
            # load positive data
            fileName = os.path.join(negPath, image)
            # print(fileName)
            img = cv.imread(fileName)
            img = cv.resize(img, (64, 128), interpolation=cv.INTER_AREA)
            # compute 3780 dimension vector hist
            hist = self.hog_extractor.compute(img, (8, 8))
            # can write to a pickle file
            for j in range(self.featureNum):
                self.featureArray[neg_index + self.PosNum, j] = hist[j]
            self.labelArray[neg_index + self.PosNum, 0] = 0
            neg_index += 1
        # print(neg_index)
        # # print(self.featureArray)
        # print("---------------")
        # print(self.labelArray)
        return self.featureArray, self.labelArray


class SVM:
    def __init__(self, featureArray=None, labelArray=None):
        self.featureArray = featureArray
        self.labelArray = labelArray
        self.svm_trainer = cv.ml.SVM_create()
    def setParam(self, type=cv.ml.SVM_C_SVC, kernel=cv.ml.SVM_LINEAR, c=0.01):
        self.svm_trainer.setType(type)
        self.svm_trainer.setKernel(kernel)
        self.svm_trainer.setC(c)
        # self.svm_trainer.setGamma(0.5)
    def train_svm(self, sampleForm=cv.ml.ROW_SAMPLE):
        return self.svm_trainer.train(self.featureArray, sampleForm, self.labelArray)
    def save_model(self):
        self.svm_trainer.save("trained_model.dat")
    def load_model(self, path):
        self.svm_trainer.load(path)


# class SK_SVM:
#     def __init__(self, pos_array, neg_array, pos_label, neg_label):
#         self.svm_trainer = sk_svm
#         self.pos_array = pos_array
#         self.neg_array = neg_array
#         self.pos_label = pos_label
#         self.neg_label = neg_label










class detector:
    def __init__(self, hog_extractor, svm, featureNum):
        self.hog_extractor = hog_extractor
        self.svm = svm
        self.featureNum = featureNum
        self.alpha = np.zeros((1), np.float32)
        self.rho = self.svm.getDecisionFunction(0, alpha=self.alpha)
        self.alphaArray = np.zeros((1, 1), np.float32)
        self.supportVArray = np.zeros((1, self.featureNum), np.float32)
        self.supportVArray = self.svm.getSupportVectors()
        self.alphaArray[0, 0] = self.alpha
        self.resultArray = np.zeros((1, self.featureNum), np.float32)
        # construct a detector
        self.detector = np.zeros((self.featureNum+1), np.float32)

    def detect(self, imagePath):
        # self.alphaArray.fill(1)
        self.resultArray = self.supportVArray
        # detect construction
        for i in range(3780):
            self.detector[i] = self.resultArray[0, i]
        self.detector[self.featureNum] = self.rho[0]

        # hog construction
        myHog = cv.HOGDescriptor()
        # myHog.setSVMDetector(np.append(sv, [[-self.rho]], 0))
        myHog.setSVMDetector(self.detector)

        files = os.listdir(imagePath)
        for image in files:
            fileName = os.path.join(imagePath, image)
            imageSrc = cv.imread(fileName)
            padding_size = int(imageSrc.shape[0]*0.3), int(imageSrc.shape[1]*0.3)
            stride_size = int(padding_size[0]*0.5), int(padding_size[1]*0.5),
            rects, wei = myHog.detectMultiScale(imageSrc, 0, winStride=stride_size, padding=padding_size, scale=1.11, finalThreshold=2)
            print(rects)
            print(wei)

            for x, y, w, h in rects:
                cv.rectangle(imageSrc, (x, y), (x + w, y + h), (255, 255, 0), thickness=2)
            # objs = myHog.detectMultiScale(imageSrc, 0, (8, 8), (32, 32), 1.05, 2)
            # print(objs)
            # x = int(objs[0][0][0])
            # y = int(objs[0][0][1])
            # w = int(objs[0][0][2])
            # h = int(objs[0][0][3])
            # cv.rectangle(imageSrc, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
            plt.imshow(imageSrc)
            plt.show()


if __name__ == "__main__":
    Hog = HOG((64, 128), (16, 16), (8, 8), (8, 8), 9)
    # ---------from----------
    processor = pre_processor(featureNum=3780, PosNum=97, NegNum=94, hog=Hog)
    featureArray, labelArray = processor.load_pos_and_neg_data(posPath="pos_set_sk", negPath="neg_set_new")
    # # --------- to ----------
    # pos_array, neg_array = featureArray[:1004], featureArray[1004:]
    # pos_label, neg_label = labelArray[:1004], labelArray[1004:]
    svm_trainer = SVM(featureArray, labelArray)
    svm_trainer.setParam(type=cv.ml.SVM_C_SVC, kernel=cv.ml.SVM_LINEAR, c=0.01)
    svm_trainer.train_svm()
    svm_trainer.save_model()
    featureArray = np.zeros((1004 + 1186, 3780), np.float32)
    labelArray = np.zeros((1004 + 1186, 1), np.int32)
    trained_svm = SVM(featureArray=featureArray, labelArray=labelArray)
    trained_svm.svm_trainer = cv.ml.SVM_load("trained_model.dat")
    myDetector = detector(Hog, featureNum=3780, svm=trained_svm.svm_trainer)

    myDetector.detect(imagePath="test_svm")
