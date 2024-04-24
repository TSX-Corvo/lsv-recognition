import numpy as np
import mediapipe as mp
import cv2
from typing import Type, Union
from keras.models import Sequential
from keras.layers import LSTM, Dense

from constants import actions
from utils import draw_styled_landmarks, extract_keypoints, mediapipe_detection

class LSVRecognition:
    model: Type[Sequential] = Sequential()
    cap: Union[Type[cv2.VideoCapture], None] = None

    def __init__(self) -> None:
        self.__define__model()
        self.__compile__model()
        self.__load_weights__()

    def __define__model(self) -> None:
        self.model.add(
            LSTM(128, return_sequences=True, activation="relu", input_shape=(29, 1662))
        )
        self.model.add(LSTM(256, return_sequences=True, activation="relu"))
        self.model.add(LSTM(128, return_sequences=False, activation="relu"))
        self.model.add(Dense(256, activation="relu"))
        self.model.add(Dense(32, activation="relu"))
        self.model.add(Dense(actions.shape[0], activation="softmax"))

    def __compile__model(self) -> None:
        self.model.compile(
            optimizer="Adam",
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"],
        )

    def __load_weights__(self):
        self.model.load_weights("model/model.h5")

    def predict(self, sequence: list):
        res = self.model.predict(np.expand_dims(sequence, axis=0))[0]
        return np.argmax(res, axis=1).tolist()

    def continuous_detection(
        self,
        source: Union[int, str],
        output,
        detection_confidence=0.5,
        wsl_compatibility=False,
    ):
        """
        Performs continuous detection on a video source and captures frames based on the detection confidence.

        Parameters:
            source (Union[int, str]): The video source, either an integer (0 for default camera) or a string (path to a video file or URL for http stream).
            output: The output.
            detection_confidence (float): The confidence threshold for detection.

        Returns:
            None
        """
        cap = cv2.VideoCapture(source)
        self.cap = cap

        # WSL compatibility trick
        if wsl_compatibility:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

        predictions = []
        sentence = []

        with mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as holistic:
            while cap.isOpened():

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                # print(results)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # 2. Prediction logic
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-29:]

                if len(sequence) == 29:
                    res = self.model.predict(np.expand_dims(sequence, axis=0))[0]
                    
                    predictions.append(np.argmax(res))

                    # Write to output
                    if np.unique(predictions[-10:])[0] == np.argmax(res):
                        if res[np.argmax(res)] > detection_confidence:

                            if len(sentence) > 0:
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                    if len(sentence) > 5:
                        sentence = sentence[-5:]


                # Break gracefully
               
            cap.release()
            cv2.destroyAllWindows()

    def stop_detection(self):
        self.cap.release()
        cv2.destroyAllWindows()    


if __name__ == "__main__":
    recognitionService = LSVRecognition()
