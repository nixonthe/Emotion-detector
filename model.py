# import libs
import numpy as np
import pandas as pd
import tensorflow as tf

# download resnet50v2
model = tf.keras.applications.ResNet50V2(include_top=False)

# define number of classes
NUM_CLASSES = 9

def build_model(model, trainable=False, show_model=True):
    """

    build model for training
    input:  model - original model
            trainable - False=freeze model
                        True=do not freeze model
            show_model -  True=show plot
                          False=hide plot
    output: new_model - model for training

    """

    # here we decide to freeze model or not
    model.trainable = trainable
    # look at model plot
    if show_model:
        model.summary()

    # define output model
    new_model = tf.keras.Sequential([
        model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    return new_model

# define model for camera.py module
resnet_model = build_model(model, True, False)
