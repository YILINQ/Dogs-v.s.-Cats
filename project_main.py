import os
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from classification import CNNclassification
from detection.hog_svm_detect import data_processor, svm, detect
dog_class = ['chihuahua', 'newfoundland', 'pug', 'saint_bernard', 'samoyed']
cat_class = ['Bengal', 'Bombay', 'Maine_Coon', 'Ragdoll', 'Russian_Blue']

dog_train_path = 'train'
dog_test_path = 'test'
dog_detect_test = 'dog_detect_test'

cat_train_path = 'train'
cat_test_path = 'test'
cat_detect_test = 'cat_detect_test'


# train -> test dog [5 types] /cat [5 types]
def classification_helper(animal_name, model, img_list):
    if animal_name == 'dog':
        my_model = CNNclassification(dog_train_path, dog_test_path)
        if model == 'smallVGG16':
            my_model.smallVGG16()
        elif model == 'traditionalCNN':
            my_model.traditionalCNN()
        # loop through the images in the directory
        # dir = os.listdir(dog_test_path)
        img_class = {}
        # for image in dir:
        #     fileName = os.path.join(dog_detect_test, image)
        #     classification = my_model.classifier(fileName)
        #     categories = classification.index(max(classification))
        #     img_class[fileName] = categories
        for image in img_list:
            classification = my_model.classifier(image)
            categories = classification.index(max(classification))
            img_class[image] = categories

        for fileName, categories in img_class:
            label = 'The object in the image is a dog in class {}'.format(dog_class[categories])
            # not quite right, transfer to image write
            plt.imshow(fileName)
            plt.title(label)

    if animal_name == 'cat':
        # classification
        my_model = CNNclassification(dog_train_path, dog_test_path)
        if model == 'smallVGG16':
            my_model.smallVGG16()
        elif model == 'traditionalCNN':
            my_model.traditionalCNN()
        # loop through the images in the directory
        # dir = os.listdir(cat_test_path)
        img_class = {}
        # for image in dir:
        #     fileName = os.path.join(cat_detect_test, image)
        #     classification = my_model.classifier(fileName)
        #     categories = classification.index(max(classification))
        #     img_class[fileName] = categories
        for image in img_list:
            classification = my_model.classifier(image)
            categories = classification.index(max(classification))
            img_class[image] = categories

        for fileName, categories in img_class:
            label = 'The object in the image is a cat in class {}'.format(cat_class[categories])
            # not quite right, transfer to image write
            plt.imshow(fileName)
            plt.title(label)


if __name__ == '__main__':
    # detection
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
    dog_imags = dog_detecor.return_detected()
    DP_cat = data_processor(posPath="cat_pos", negPath="cat_neg", hog=hog_extractor)
    DP_cat.load_data()
    pos_data, neg_data, pos_label, neg_label, unscale = DP_cat.transform_data()
    svm_trainer_cat = svm(pos_data, neg_data, pos_label, neg_label, unscale)
    svm_trainer_cat.train_svm()
    cat_detecor = detect(svm_trainer_cat, image_path="test_svm_cat", hog=hog_extractor, grouped=False)
    cat_detecor.detect_window()
    cat_detecor = detect(svm_trainer_cat, image_path="test_svm_cat", hog=hog_extractor, grouped=True)
    cat_detecor.detect_window()
    cat_images = cat_detecor.return_detected()
    # classification
    classification_helper('dog', 'smallVGG16', dog_imags)
    classification_helper('dog', 'traditionalCNN', cat_images)

    classification_helper('cat', 'smallVGG16')

