# import torch
import time
from functools import wraps


def start():
    print("starting...")

    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras import datasets, layers, models
    from tensorflow.keras.models import load_model

    from sklearn.model_selection import train_test_split
    import os
    import shutil
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    import glob
    from .dataloader import DataLoader
    from .cnn_model import Model
    from .preprocessing import move_files

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    img_width, img_height = 180, 180
    BATCH_SIZE = 25

    main_dir = "/Users/sebastianblum/GitHub/ML_UseCases/chest_xray/data/"
    train_data_dir = f"{main_dir}train/"
    validation_data_dir = f"{main_dir}val/"
    test_data_dir = f"{main_dir}test/"
    main_dir_2 = "/Users/sebastianblum/GitHub/ML_UseCases/chest_xray/data/new/"
    train_data_dir_2 = f"{main_dir_2}train/"
    validation_data_dir_2 = f"{main_dir_2}val/"

    # this is done as val set is so low
    filenames = tf.io.gfile.glob(f"{train_data_dir}*/*")
    filenames.extend(tf.io.gfile.glob(f"{validation_data_dir}*/*"))
    # re-split filenames to have a new train/test ratio
    train_filenames, val_filenames = train_test_split(filenames, test_size=0.2)

    def data_count(classtype, filenames):
        if classtype == 0:
            return sum(map(lambda x: "NORMAL" in x, filenames))
        elif classtype == 1:
            return sum(map(lambda x: "PNEUMONIA" in x, filenames))

    print(f"training data | Normal : {data_count(0, train_filenames)}")
    print(f"training data | PNEUMONIA : {data_count(1, train_filenames)}")
    print(f"validation data | Normal : {data_count(0, val_filenames)}")
    print(f"validation data | PNEUMONIA : {data_count(1, val_filenames)}")
    # have to set weights for training as less normales then pneumonia
    # print(train_filenames[0])

    TRAIN_IMG_COUNT = len(train_filenames)
    VAL_IMG_COUNT = len(val_filenames)
    COUNT_NORMAL = data_count(0, train_filenames)
    COUNT_PNEUMONIA = data_count(1, train_filenames)

    def preprocessing():
        move_files(train_filenames, train_data_dir_2)
        move_files(val_filenames, validation_data_dir_2)

    def create_input():
        # Image Augmentation to have more data samples using zoom, flip and shear
        # done for large datasets. this one could actually be done by loading everything into memory
        datagen_train = ImageDataGenerator(
            rescale=1.0 / 255, zoom_range=0.2, shear_range=0.2, horizontal_flip=True
        )
        datagen_val = ImageDataGenerator(rescale=1.0 / 255)
        datagen_test = ImageDataGenerator(rescale=1.0 / 255)
        train_generator = DataLoader.get_generator(
            datagen_train, train_data_dir_2, img_width, img_height, BATCH_SIZE
        )
        validation_generator = DataLoader.get_generator(
            datagen_val, validation_data_dir_2, img_width, img_height, BATCH_SIZE
        )
        test_generator = DataLoader.get_generator(
            datagen=datagen_test,
            directory=test_data_dir,
            img_width=180,
            img_height=180,
            BATCH_SIZE=25,
        )
        image_batch, label_batch = next(iter(train_generator))
        DataLoader.show_batch(image_batch, label_batch, BATCH_SIZE)

    class_weight = DataLoader.class_weights(
        COUNT_PNEUMONIA, COUNT_NORMAL, TRAIN_IMG_COUNT
    )
    print(f"Weight for class 0: {class_weight.get(0)}")
    print(f"Weight for class 1: {class_weight.get(1)}")

    input_shape = image_batch[1].shape
    print(f"shape: {input_shape}")
    output_shape = 1

    cnn = Model.make_model(input_shape, output_shape)
    print(cnn.summary())

    def train_model():
        EPOCHS = 20
        METRICS = [
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
        cnn.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=METRICS)
        history = cnn.fit(
            train_generator,
            steps_per_epoch=TRAIN_IMG_COUNT // BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=validation_generator,
            validation_steps=VAL_IMG_COUNT // BATCH_SIZE,
            class_weight=class_weight,
        )
        Model.make_graph(history, ["precision", "recall", "accuracy", "loss"])
        cnn.save("chestxray_cnn_model_3.h5")

    def predict():
        cnn = load_model("chestxray_cnn_model_3.h5")
        loss, acc, prec, rec = cnn.evaluate(test_generator)
        print("loss: {:.2f}".format(loss))
        print("acc:  {:.2f}".format(acc))
        print("prec: {:.2f}".format(prec))
        print("rec:  {:.2f}".format(rec))
