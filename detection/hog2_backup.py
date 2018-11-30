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

# set up
PosNum = 8004
NegNum = 349
winSize = (64, 64)
block_size = (16, 16)  # 105
blockStride = (8, 8)
cellSize = (8, 8)  # 4 cell
nBins = 9  # 9 bin 3780 dimension vector
hog_extractor = cv.HOGDescriptor(winSize, block_size, blockStride, cellSize, nBins)

# ------ svm ------- #
# ------ svm ------- #

# init data array
featureNum = 3780
pos_array = []
neg_array = []
pos_label = []
neg_label = []
# process data array
posPath = "pos_set_sk"
negPath = "neg_set_new"
posFiles = os.listdir(posPath)
for image in posFiles:
    # load positive data
    fileName = os.path.join(posPath, image)
    # print(fileName)
    img = cv.imread(fileName)
    img = cv.resize(img, (64, 128), interpolation=cv.INTER_AREA)
    # compute 3780 dimension vector hist
    hist = hog_extractor.compute(img, (8, 8))
    # can write to a pickle file
    pos_array.append(hist.reshape(-1, 15876)[0])
    pos_label.append(1)

negFiles = os.listdir(negPath)
counter = 0
for image in negFiles:
    if counter < 900:
    # load positive data
        fileName = os.path.join(negPath, image)
        # print(fileName)
        img = cv.imread(fileName)
        img = cv.resize(img, (64, 128), interpolation=cv.INTER_AREA)
        # compute 3780 dimension vector hist
        hist = hog_extractor.compute(img, (8, 8))
        # can write to a pickle file
        neg_array.append(hist.reshape(-1, 15876)[0])
        neg_label.append(0)
        counter += 1

pos_data = np.asarray(pos_array)
neg_data = np.asarray(neg_array)
unscale = np.vstack((pos_data, neg_data)).astype(np.float64)
scaler = StandardScaler().fit(unscale)

X = scaler.transform(unscale)
Y = np.asarray(pos_label + neg_label)

train_data, test_data, train_label, test_label = train_test_split(X, Y, test_size=0.2,
                                                                  random_state=0)
# linear_svc = LinearSVC()
# linear_svc.fit(train_data, train_label)
svc = LinearSVC()
svc.fit(train_data, train_label)
# with open("sk_svc.pickle") as handle:
#     pickle.dump(svc, handle, protocol=pickle.HIGHEST_PROTOCOL)
score = svc.score(test_data, test_label)
print("Accuracy:" + str(score*100.0) + "%")

# testing
posFiles = os.listdir("test_image")
test = []
for image in posFiles:
    fileName = os.path.join("test_image", image)
    # print(fileName)
    img = cv.imread(fileName)

    recs = []
    winSize = (int(img.shape[0] * 0.5), int(img.shape[1] * 0.5))
    stride = (int(winSize[0]*0.2), int(winSize[1] * 0.2))
    for x_ in range(winSize[0], img.shape[1], stride[0]):
        for y_ in range(winSize[1], img.shape[0], stride[1]):
            x = (x_ - winSize[0], x_)
            y = (y_ - winSize[1], y_)
            # img_ = np.zeros(winSize)
            a = img[y[0]:y[1], x[0]:x[1], :]
            img_ = cv.resize(a, (64, 128), interpolation=cv.INTER_AREA)
            new_hist = hog_extractor.compute(img_, (8, 8))
            test = [new_hist.reshape(-1, 15876)[0]]
            if svc.predict(test) == [1]:
                recs.append((x[0], y[0], x[1], y[1]))
            print(recs)
    for rec in recs:
        x = rec[0]
        x_ = rec[2]
        y = rec[1]
        y_ = rec[3]
        cv.rectangle(img, (x, y), (x_, y_), (255, 255, 0), thickness=2)
    plt.imshow(img)
    plt.show()
