import cv2
import matplotlib.pyplot as plt
from copy import deepcopy
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from train_utils import build_model
from params import IMG_SHAPE


def capture_video():
    """

    function for capture video
    output: frame - video's shot

    """
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ret, frame = cam.read()
        cv2.imshow("facial emotion recognition", frame)
        # press s to save a certain shot and stop recording
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite('image.jpg', frame)
            break

    return frame


def load_face_detector(config):
    """
    
    define face detector with loaded configuration file
    input:  config - path to configuration file
    output: face_detector - loaded detector

    """

    return cv2.CascadeClassifier(config)


def get_bb(grayscale_image, original_image, emotion, face_detector, save_image=True):
    """

    creating bounding box for image
    input:  grayscale_image - image for detector
            original_image - input image
            emotion - predicted emotion
            face_detector - loaded detector

    """

    faces = face_detector.detectMultiScale(grayscale_image, 1.3, 5)
    one_face = faces[0]
    x, y, w, h = one_face
    rgb_image_with_boundingbox = deepcopy(original_image)
    rgb_image_with_boundingbox = cv2.rectangle(
        rgb_image_with_boundingbox,
        (y, x),
        (y + h, x + w),
        (0, 255, 0),
        3
    )
    rgb_image_with_boundingbox_and_text = deepcopy(rgb_image_with_boundingbox)
    rgb_image_with_boundingbox_and_text = cv2.putText(
        rgb_image_with_boundingbox_and_text,
        emotion,
        (y, x - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )
    plt.figure(figsize=(15, 5))
    plt.imshow(rgb_image_with_boundingbox_and_text)
    plt.show()

    if save_image:
        plt.imsave('photo.jpg', rgb_image_with_boundingbox_and_text)


class Decemo:
    def __init__(self, weights, image, rgb=True):
        self.weights = weights
        self.image = image
        self.rgb = rgb

    def predict(self):
        model = tf.keras.applications.ResNet50V2(include_top=False)
        resnet_model = build_model(model, True, False)
        resnet_model.load_weights(self.weights)

        if self.rgb:
            image = plt.imread(self.image)
            x = cv2.resize(image, (IMG_SHAPE, IMG_SHAPE))
        else:
            x = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        p_image = preprocess_input(x)
        input_image = p_image[None, ...]
        predict = resnet_model.predict_classes(input_image)

        emo = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise', 'uncertain']

        return emo[int(predict)]

    def show_predict(self, emotion, face_detector):
        if self.rgb:
            rgb_image = plt.imread(self.image)
            grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            get_bb(grayscale_image, rgb_image, emotion, face_detector)
        else:
            rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            get_bb(grayscale_image, rgb_image, emotion, face_detector)
