import cv2
import matplotlib.pyplot as plt
from copy import deepcopy
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from model import resnet_model

PATH_TO_WEIGHTS = 'weights/model_6'
PATH_TO_DETECTOR = 'haarcascade_frontalface_default.xml'


def capture_video():
    """

    function for capture video
    output: frame - video's shot

    """
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ret, frame = cam.read()
        cv2.imshow("facial emotion recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite('image.jpg', frame)
            break

    return frame


def load_face_detector(config):
    """

    input:  config - path to configuration file
    output: face_detector - loaded detector

    """

    return cv2.CascadeClassifier(config)


def get_bb(grayscale_image, original_image, emotion, face_detector, save_image=True):
    """

    creating bounding box for image
    input:  grayscale_image - image for detector
            original_image - input image

    """

    # detect face
    faces = face_detector.detectMultiScale(grayscale_image, 1.3, 5)
    one_face = faces[0]
    x, y, w, h = one_face
    # make copy
    rgb_image_with_boundingbox = deepcopy(original_image)
    # create bb
    rgb_image_with_boundingbox = cv2.rectangle(
        rgb_image_with_boundingbox,
        (y, x),
        (y + h, x + w),
        (0, 255, 0),
        3
    )
    # make copy
    rgb_image_with_boundingbox_and_text = deepcopy(rgb_image_with_boundingbox)
    # add text
    rgb_image_with_boundingbox_and_text = cv2.putText(
        rgb_image_with_boundingbox_and_text,
        emotion,
        (y, x - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )
    # plot image
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
        # load weights
        resnet_model.load_weights(self.weights)

        if self.rgb:
            image = plt.imread(self.image)
            x = cv2.resize(image, (224, 224))
        else:
            x = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        p_image = preprocess_input(x)
        input_image = p_image[None, ...]
        predict = resnet_model.predict_classes(input_image)

        emo = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise', 'uncertain']

        return emo[int(predict)]

    def show_predict(self, emotion, face_detector, save_image=True):
        if self.rgb:
            rgb_image = plt.imread(self.image)
            grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            get_bb(grayscale_image, rgb_image, emotion, face_detector)
        else:
            rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            get_bb(grayscale_image, rgb_image, emotion, face_detector)


# frame = capture_video()
face_detector = load_face_detector(PATH_TO_DETECTOR)
dec = Decemo(PATH_TO_WEIGHTS, 'image.jpg')
emotion = dec.predict()
dec.show_predict(emotion, face_detector)