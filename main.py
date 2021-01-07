import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd

import os
import random
import shutil
import pathlib

from keras_preprocessing.image import ImageDataGenerator

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def create_folder_structure():
    shutil.rmtree('./train')
    pathlib.Path("./train/train/dogs").mkdir(parents=True, exist_ok=True)
    pathlib.Path("./train/train/cats").mkdir(parents=True, exist_ok=True)
    pathlib.Path("./train/validate/dogs").mkdir(parents=True, exist_ok=True)
    pathlib.Path("./train/validate/cats").mkdir(parents=True, exist_ok=True)


def copy_images(source_list, destination_path):
    for image in source_list:
        shutil.copyfile(f'./input_data/{image}', f'./train/{destination_path}/{image}')


def create_training_and_validation_set(train_validation_split):
    cat_and_dog_images = os.listdir('./input_data')

    cat_images = list(filter(lambda image: 'cat' in image, cat_and_dog_images))
    dog_images = list(filter(lambda image: 'dog' in image, cat_and_dog_images))

    random.shuffle(cat_images)
    random.shuffle(dog_images)

    split_index = int(len(cat_images) * train_validation_split)

    training_cats = cat_images[:split_index]
    validation_cats = cat_images[split_index:]
    training_dogs = dog_images[:split_index]
    validation_dogs = dog_images[split_index:]

    create_folder_structure()
    copy_images(training_dogs, 'train/dogs')
    copy_images(training_cats, 'train/cats')
    copy_images(validation_dogs, 'validate/dogs')
    copy_images(validation_cats, 'validate/cats')


def train_model():
    train_gen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_iterator = train_gen.flow_from_directory('./train/train',
                                                   target_size=(150, 150),
                                                   batch_size=20,
                                                   class_mode='binary')

    validation_gen = ImageDataGenerator(rescale=1. / 255.0)
    validation_iterator = validation_gen.flow_from_directory('./train/validate',
                                                             target_size=(150, 150),
                                                             batch_size=10,
                                                             class_mode='binary')

    model = keras.models.Sequential([
        keras.layers.Conv2D(16, (3, 3), activation=tf.nn.relu, input_shape=(150, 150, 3)),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(units=512, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    model.compile(optimizer=tf.optimizers.Adam(),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['accuracy'])

    history = model.fit(train_iterator,
                        validation_data=validation_iterator,
                        steps_per_epoch=1000,
                        epochs=50,
                        validation_steps=500)

    model.save('dogs-vs-cats.h5')

    return history


def plot_result(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def load_and_predict():
    model = keras.models.load_model('dogs-vs-cats.h5')

    test_generator = ImageDataGenerator(rescale=1. / 255)

    test_iterator = test_generator.flow_from_directory(
        './input_test',
        target_size=(150, 150),
        shuffle=False,
        class_mode='binary',
        batch_size=1)

    ids = []
    for filename in test_iterator.filenames:
        ids.append(int(filename.split('\\')[1].split('.')[0]))

    predict_result = model.predict(test_iterator, steps=len(test_iterator.filenames))
    predictions = []
    for index, prediction in enumerate(predict_result):
        predictions.append([ids[index], prediction[0]])
    predictions.sort()

    return predictions


create_training_and_validation_set(0.8)
result_history = train_model()
plot_result(result_history)
predictions = load_and_predict()
df = pd.DataFrame(data=predictions, index=range(1, 12501), columns=['id', 'label'])
df = df.set_index(['id'])
df.to_csv('submission.csv')