import tensorflow as tf
import tensorflow.keras as keras

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

import os
import random
import shutil
from shutil import copyfile
from os import path
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_training_validation_folders(distribution):
    images = os.listdir('./data/train_input/train/')
    cats = list(filter(lambda image: 'cat' in image, images))
    dogs = list(filter(lambda image: 'dog' in image, images))

    random.shuffle(cats)
    random.shuffle(dogs)

    split_index = int(len(dogs) * distribution)
    training_dogs = dogs[0:split_index]
    validation_dogs = dogs[split_index:]
    training_cats = cats[0:split_index]
    validation_cats = cats[split_index:]

    print(f'Cats: {len(cats)}, Dogs {len(dogs)}, total={len(images)}')
    print(f'Cats train: {len(training_cats)}, validation {len(validation_cats)}')
    print(f'Dogs train: {len(training_dogs)}, validation {len(validation_dogs)}')

    if path.exists('./data/training/'):
        shutil.rmtree('./data/training')

    os.mkdir('./data/training')
    os.mkdir('./data/training/train')
    os.mkdir('./data/training/train/dogs')
    os.mkdir('./data/training/train/cats')
    os.mkdir('./data/training/validate')
    os.mkdir('./data/training/validate/dogs')
    os.mkdir('./data/training/validate/cats')

    for dog in training_dogs:
        copyfile(f'./data/train_input/train/{dog}', f'./data/training/train/dogs/{dog}')
    for dog in validation_dogs:
        copyfile(f'./data/train_input/train/{dog}', f'./data/training/validate/dogs/{dog}')
    for cat in training_cats:
        copyfile(f'./data/train_input/train/{cat}', f'./data/training/train/cats/{cat}')
    for cat in validation_cats:
        copyfile(f'./data/train_input/train/{cat}', f'./data/training/validate/cats/{cat}')

    print(f'Training dogs: {len(os.listdir("./data/training/train/dogs"))}')
    print(f'Validating dogs: {len(os.listdir("./data/training/validate/dogs"))}')

    print(f'Training cats: {len(os.listdir("./data/training/train/cats"))}')
    print(f'Validating cats: {len(os.listdir("./data/training/validate/cats"))}')


#create_training_validation_folders(0.8)

train_generator = ImageDataGenerator(rescale=1. / 255,
                                     rotation_range=40,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     fill_mode='nearest')

train_stream = train_generator.flow_from_directory('./data/training/train/',
                                                   target_size=(150, 150),
                                                   batch_size=10,
                                                   class_mode='binary')

validation_generator = ImageDataGenerator(rescale=1. / 255)
validation_stream = validation_generator.flow_from_directory('./data/training/validate/',
                                                             target_size=(150, 150),
                                                             batch_size=10,
                                                             class_mode='binary')

model = keras.models.Sequential([
    keras.layers.Conv2D(16, (3, 3), activation=tf.nn.relu, input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.binary_crossentropy, metrics=['accuracy'])

history = model.fit(train_stream, epochs=10, validation_data=validation_stream)

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Accuracy (training data)')
plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
plt.title('Accuracy')
plt.ylabel('Accuracy value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()
