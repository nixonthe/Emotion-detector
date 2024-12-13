import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_input_resnet
from train_utils import make_aug_train_gen, build_model, training
from params import IMG_SHAPE, output_model_filename


def main():
    train_dir = Path('/data/train/')

    df = pd.read_csv('data/train.csv')

    resnet_model = tf.keras.applications.ResNet50V2(include_top=False)

    aug_train_gen = make_aug_train_gen(
        preprocess_func=preprocess_input_resnet, directory=train_dir, input_dataframe=df, img_shape=IMG_SHAPE
    )
    model = build_model(model=resnet_model, trainable=True, show_model=False)
    print('---------------------------------------------------------')
    print('Training...')
    training(model, aug_train_gen)
    print('----------------------------------------------------------')
    print('Saving model...')
    model.save_weights(output_model_filename)


if __name__  == "__main__":
    main()
