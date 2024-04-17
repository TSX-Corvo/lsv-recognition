from typing import Type
import numpy as np


from keras.models import Sequential
from keras.layers import LSTM, Dense

from constants import actions

class LSVRecognition:
    model: Type[Sequential] = Sequential()

    def __init__(self) -> None:
        self.__define__model()
        self.__compile__model()
        self.__load_weights__()

    def __define__model(self) -> None:
        self.model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(29,1662)))
        self.model.add(LSTM(256, return_sequences=True, activation='relu'))
        self.model.add(LSTM(128, return_sequences=False, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(actions.shape[0], activation='softmax'))

    def __compile__model(self) -> None:
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    def __load_weights__(self):
        self.model.load_weights('models/model.h5')

    def predict(self, sequence: list):
        res = self.model.predict(np.expand_dims(sequence, axis=0))[0]
        return np.argmax(res, axis=1).tolist()

if __name__ == '__main__':
    pass