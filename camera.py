# import libs
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from model import resnet_model

# define paths to weights and face model configuration
PATH_TO_WEIGHTS = 'weights/model_6'
PATH_TO_DETECTOR = 'haarcascade_frontalface_default.xml'


def capture_video():
    """

    function for capture video
    output: frame - video's shot

    """
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # start recordering video
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

    # detect face
    faces = face_detector.detectMultiScale(grayscale_image, 1.3, 5)
    one_face = faces[0]
    # get coordinates for bb
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
    # define size of image
    plt.figure(figsize=(15, 5))
    # load image
    plt.imshow(rgb_image_with_boundingbox_and_text)
    # plot image
    plt.show()
    
    # if needed you can save it
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
        
        # in case of rbg image
        if self.rgb:
            image = plt.imread(self.image)
            x = cv2.resize(image, (224, 224))
        # in case of brg image
        else:
            x = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        # preprocess image for resnet model
        p_image = preprocess_input(x)
        # add batch
        input_image = p_image[None, ...]
        # predict emotion
        predict = resnet_model.predict_classes(input_image)
        
        # list of emotion in alphabetical order
        emo = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise', 'uncertain']

        return emo[int(predict)]

    def show_predict(self, emotion, face_detector, save_image=True):
        # in case of rgb image
        if self.rgb:
            # read image
            rgb_image = plt.imread(self.image)
            # convert it to gray for face detector
            grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            # create bb and add prediction label on photo
            get_bb(grayscale_image, rgb_image, emotion, face_detector)
        # in case of bgr image
        else:
            # convert from bgr to rgb
            rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            # convert it to gray for face detector
            grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            # create bb and add prediction label on photo
            get_bb(grayscale_image, rgb_image, emotion, face_detector)
