import pickle
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.optimizers import SGD, Adam
from keras import optimizers
import numpy as np
from keras.applications import VGG19


"""
Parameters
"""
img_width, img_height = 64, 64
batch_size = 32
samples_per_epoch = 2000
epochs = 30
validation_steps = 200
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 2
pool_size = 2
classes_num = 5
lr = 0.0004

classifier = Sequential()
classifier.add(Convolution2D(32, kernel_size=(3, 3),padding='same',input_shape=(64, 64, 3)))
classifier.add(Activation('relu'))
classifier.add(Convolution2D(64, (3, 3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Convolution2D(64,(3, 3), padding='same'))
classifier.add(Activation('relu'))
classifier.add(Convolution2D(64, 3, 3))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Flatten())
classifier.add(Dense(512))
classifier.add(Activation('relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(5))
classifier.add(Activation('softmax'))
classifier.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)


train_generator = train_datagen.flow_from_directory(
    'dog_cat_dataset/cat/train',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    'dog_cat_dataset/cat/test',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

classifier.fit_generator(
    train_generator,
    samples_per_epoch = samples_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps)

