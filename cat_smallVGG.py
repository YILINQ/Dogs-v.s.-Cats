"""
This is a draft version of smallVGG16 for cat, only for testing
"""


import pickle
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD


img_width, img_height = 64, 64
batch_size = 16
samples_per_epoch = 2500
epochs = 20
validation_steps = 300
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 2
pool_size = 2
classes_num = 5
lr = 0.0004

model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()

input = Input(shape=(64,64,3),name = 'image_input')

output_vgg16_conv = model_vgg16_conv(input)

x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(512, activation='relu', name='fc1')(x)
x = Dense(512, activation='relu', name='fc2')(x)
x = Dense(5, activation='softmax', name='predictions')(x)

my_model = Model(input=input, output=x)

my_model.summary()

sdg = SGD(lr=0.001, clipnorm=1.)
my_model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
  "dog_cat_dataset/cat/train",
  target_size=(img_height, img_width),
  batch_size=batch_size,
  class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
  "dog_cat_dataset/cat/test",
  target_size=(img_height, img_width),
  batch_size=batch_size,
  class_mode='categorical')

my_model.fit_generator(
  train_generator,
  samples_per_epoch = samples_per_epoch,
  epochs=epochs,
  validation_data=validation_generator,
  validation_steps=validation_steps)


with open('classifier.pickle', 'wb') as handle:
    pickle.dump(my_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
