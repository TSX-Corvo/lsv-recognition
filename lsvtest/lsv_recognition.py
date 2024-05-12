import numpy as np
import numpy.typing as npt
import mediapipe as mp
import cv2
from typing import Type, Union, Callable
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense
from importlib import resources

from .constants import actions
from .utils import extract_keypoints, mediapipe_detection, draw_styled_landmarks

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
        with resources.path('lsvxd.model', 'model.h5') as weights_path:
            self.model.load_weights(str(weights_path))

    def predict(self, sequence: list):
        res = self.model.predict(np.expand_dims(sequence, axis=0))[0]
        return np.argmax(res, axis=1).tolist()
    
    def __wsl_compatibility__(self, cap: Type[cv2.VideoCapture]):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

    def extract_hands_keypoints(self, keypoints: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return keypoints[1536:1662]

    def continuous_detection(
        self,
        source: Union[int, str],
        output: Callable[[str], None],
        detection_confidence=0.5,
        wsl_compatibility=False,
        holistic_model = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ),
        debug_mode=False
    ):
        """
        Performs continuous detection on a video source and captures frames based on the detection confidence.

        Parameters:
            source: The video source, either an integer (0 for default camera) or a string (path to a video file or URL for http stream).
            output: Function executed when a gesture is detected, it receives the recognized gesture as parameter.
            detection_confidence (float): The confidence threshold for detection.

        Returns:
            None
        """
        cap = cv2.VideoCapture(source)
        self.cap = cap

        # WSL compatibility trick
        if wsl_compatibility:
           self.__wsl_compatibility__(cap)

        # State variables
        predictions = []
        sentence = []
        sequence = []

        prev_hands = np.zeros(42*3)

        frames_with_changes = 0

        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            if not ret:
                continue

            # Make detections
            image, results = mediapipe_detection(frame, holistic_model)

            if debug_mode:
                # Draw landmarks
                draw_styled_landmarks(image, results)

            #Prediction logic
            keypoints = extract_keypoints(results)


            hands_keypoints = self.extract_hands_keypoints(keypoints)

            diff_l = np.subtract(hands_keypoints[0:62], prev_hands[0:62])
            diff_r = np.subtract(hands_keypoints[63:126], prev_hands[63:126])
            percent_l = diff_l / prev_hands[0:62]
            percent_r = diff_r / prev_hands[63:126]

            avg_l = np.mean(percent_l, axis=0)
            avg_r = np.mean(percent_r, axis=0)
            
            if abs(avg_l) > 0.05 or abs(avg_r) > 0.05:
                frames_with_changes = min(10, frames_with_changes + 1)
            else:
                frames_with_changes = max(0, frames_with_changes - 1)
            
            # print(f"{frames_with_changes} frames with changes")

            prev_hands = hands_keypoints

            sequence.append(keypoints)
            sequence = sequence[-29:]

            if len(sequence) == 29:
                res = self.model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                
                max_prob = np.argmax(res)
                
                predictions.append(max_prob)

                # Write to output
                if np.unique(predictions[-20:])[0] == max_prob:
                    if res[max_prob] > detection_confidence:

                        detected = actions[max_prob]

                        if len(sentence) > 0:
                            if detected != sentence[-1] and frames_with_changes > 1:
                                sentence.append(detected)                              
                                output(f"{detected}")
                        else:
                            sentence.append(detected)
                            # output(detected)

                if len(sentence) > 5:
                    sentence = sentence[-5:]


            # Show to screen
            if debug_mode:
                cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()

    def stop_detection(self):
        self.cap.release()
        cv2.destroyAllWindows()    


if __name__ == "__main__":
    recognition_service = LSVRecognition()

    recognition_service.continuous_detection(
        source=0, 
        output=lambda token: print(token), 
        wsl_compatibility=True, 
        debug_mode=True
    )
