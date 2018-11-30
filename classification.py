import pickle
from keras import optimizers
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import numpy as np


class CNNclassification:

    def __init__(self, train_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.my_model = None
        self.history = None
        self.pickel_file = None

    def smallVGG16(self):
        img_width, img_height = 64, 64
        batch_size = 16
        samples_per_epoch = 2500
        epochs = 30
        validation_steps = 300
        lr = 0.01

        model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
        model_vgg16_conv.summary()

        input = Input(shape=(64, 64, 3), name='image_input' )
        output_vgg16_conv = model_vgg16_conv(input)

        x = Flatten(name='flatten')(output_vgg16_conv)
        x = Dense(512, activation='relu', name='fc1')(x)
        x = Dense(512, activation='relu', name='fc2')(x)
        x = Dense(5, activation='softmax', name='predictions')(x)

        my_model = Model(input=input, output=x)

        my_model.summary()

        sdg = SGD(lr=lr, clipnorm=1.)
        my_model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            self.train_dataset,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical')

        validation_generator = test_datagen.flow_from_directory(
            self.test_dataset,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical')

        history = my_model.fit_generator(
            train_generator,
            samples_per_epoch=samples_per_epoch,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps)

        self.my_model = my_model
        self.history = history

        with open('SMALLVGGclassifier.pickle', 'wb') as handle:
            pickle.dump(self.my_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.pickel_file = 'SMALLVGGclassifier.pickle'

    def traditionalCNN(self):
        img_width, img_height = 64, 64
        batch_size = 16
        samples_per_epoch = 2500
        epochs = 30
        validation_steps = 300
        lr = 0.0004

        my_model = Sequential()
        my_model.add(Convolution2D(32, kernel_size=(3, 3), padding='same', input_shape=(64, 64, 3)))
        my_model.add(Activation('relu'))
        my_model.add(Convolution2D(64, (3, 3)))
        my_model.add(Activation('relu'))
        my_model.add(MaxPooling2D(pool_size=(2, 2)))
        my_model.add(Dropout(0.25))

        my_model.add(Convolution2D(64, (3, 3), padding='same'))
        my_model.add(Activation('relu'))
        my_model.add(Convolution2D(64, 3, 3))
        my_model.add(Activation('relu'))
        my_model.add(MaxPooling2D(pool_size=(2, 2)))
        my_model.add(Dropout(0.25))

        my_model.add(Flatten())
        my_model.add(Dense(512))
        my_model.add(Activation('relu'))
        my_model.add(Dropout(0.5))
        my_model.add(Dense(5))
        my_model.add(Activation('softmax'))

        my_model.compile(loss='categorical_crossentropy',
                         optimizer=optimizers.RMSprop(lr=lr),
                         metrics=['accuracy'])

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            self.train_dataset,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical')

        validation_generator = test_datagen.flow_from_directory(
            self.test_dataset,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical')

        history = my_model.fit_generator(
            train_generator,
            samples_per_epoch=samples_per_epoch,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps)

        self.my_model = my_model
        self.history = history

        with open('TraditionalCNNclassifier.pickle', 'wb') as handle:
            pickle.dump(self.my_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.pickel_file = 'TraditionalCNNclassifier.pickle'

    def classify(self, path_to_img):
        """
        path_to_ima is the path to the test input image that we write the data in.
        """
        with open(self.pickel_file, 'rb') as handle:
            classifier = pickle.load(handle)

        test_image = image.load_img(path_to_img, target_size=(64, 64))

        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = classifier.predict(test_image)

        return result   # list of 5 score to the 5 classes, we need to see the max one
