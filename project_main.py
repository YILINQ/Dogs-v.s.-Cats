import os
import cv2 as cv
import matplotlib.pyplot as plt

from classification import CNNclassification

dog_class = ['chihuahua', 'newfoundland', 'pug', 'saint_bernard', 'samoyed']
cat_class = ['Bengal', 'Bombay', 'Maine_Coon', 'Ragdoll', 'Russian_Blue']

dog_train_path = 'train'
dog_test_path = 'test'
dog_detect_test = 'dog_detect_test'

cat_train_path = 'train'
cat_test_path = 'test'
cat_detect_test = 'cat_detect_test'


# train -> test dog [5 types] /cat [5 types]
def classification_helper(animal_name, model):
    if animal_name == 'dog':
        my_model = CNNclassification(dog_train_path, dog_test_path)
        if model == 'smallVGG16':
            my_model.smallVGG16()
        elif model == 'traditionalCNN':
            my_model.traditionalCNN()
        # loop through the images in the directory
        dir = os.listdir(dog_test_path)
        img_class = {}
        for image in dir:
            fileName = os.path.join(dog_detect_test, image)
            classification = my_model.classifier(fileName)
            categories = classification.index(max(classification))
            img_class[fileName] = categories

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
        dir = os.listdir(cat_test_path)
        img_class = {}
        for image in dir:
            fileName = os.path.join(cat_detect_test, image)
            classification = my_model.classifier(fileName)
            categories = classification.index(max(classification))
            img_class[fileName] = categories

        for fileName, categories in img_class:
            label = 'The object in the image is a cat in class {}'.format(cat_class[categories])
            # not quite right, transfer to image write
            plt.imshow(fileName)
            plt.title(label)


if __name__ == '__main__':
    # detection

    # classification
    classification_helper('dog', 'smallVGG16')
    classification_helper('dog', 'traditionalCNN')

    classification_helper('cat', 'smallVGG16')

