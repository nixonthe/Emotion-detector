import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from params import NUM_CLASSES


def make_aug_train_gen(preprocess_func, directory, img_shape, input_dataframe):
    """

    augment dataset for training
    input:  preprocess_func - preprocess function of model
          directory - path to train data
          img_shape - image shape for model input
    output: aug_train_gen - complete generator for training

    """

    aug_gen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      vertical_flip=True,
      preprocessing_function=preprocess_func
    )

    aug_train_gen = aug_gen.flow_from_dataframe(
      dataframe=input_dataframe,
      directory=directory,
      x_col='image_path',
      y_col='emotion',
      target_size=(img_shape, img_shape)
    )

    return aug_train_gen


def training(model, train_data, optimizer=tf.keras.optimizers.Adam(),
             loss=tf.keras.losses.categorical_crossentropy, epochs=20):
    """

    train certain model
    input:  model - trained model
            optimizer - optimizer
            train_data - data for training
            loss - loss
            epochs - number of epochs

    """

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=tf.keras.metrics.CategoricalAccuracy()
                  )

    model.fit_generator(
        generator=train_data,
        epochs=epochs,
    )


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

    model.trainable = trainable
    if show_model:
        model.summary()

    new_model = tf.keras.Sequential([
        model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    return new_model
