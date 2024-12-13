from camera import Decemo, load_face_detector
from params import PATH_TO_WEIGHTS, PATH_TO_DETECTOR, path_to_image


def main():
    camera = Decemo(weights=PATH_TO_WEIGHTS, image=path_to_image)
    predicted_emo = camera.predict()
    camera.show_predict(emotion=predicted_emo, face_detector=load_face_detector(PATH_TO_DETECTOR))

if __name__ == "__main__":
    main()
