import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import mediapipe as mp

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

class LSVRecognition:
    model: Sequential

    def __init__(self):
        self.model = Sequential()
        self.model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(29,1662)))
        self.model.add(LSTM(256, return_sequences=True, activation='relu'))
        self.model.add(LSTM(128, return_sequences=False, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))

if __name__ == '__main__':
    pass