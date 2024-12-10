import librosa
import numpy as np
import keras
from keras import layers

from flask import Flask, request, redirect, render_template_string, send_from_directory
from werkzeug.utils import secure_filename
import os
from io import BytesIO
import base64

''' Class to Make Predicitions '''

class LivePredictions:
    def __init__(self):
        self.path = ".//model//Emotion_Voice_Detection_Model.h5"

        input_layer = keras.Input(shape=(40, 1))
        x = layers.Conv1D(64, 5, padding='same')(input_layer)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(8)(x)
        output_layer = layers.Activation('softmax')(x)
        self.model = keras.Model(inputs=input_layer, outputs=output_layer)
        self.model.load_weights(self.path)

    @staticmethod
    def convert_class_to_emotion(pred):
        label_conversion = {'0': 'neutral',
                            '1': 'calm',
                            '2': 'happy',
                            '3': 'sad',
                            '4': 'angry',
                            '5': 'fearful',
                            '6': 'disgust',
                            '7': 'surprised'}

        for key, value in label_conversion.items():
            if int(key) == pred:
                label = value

        return label

    def make_predictions(self, file):
        data, sampling_rate = librosa.load(file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, axis=0)
        predictions = self.model.predict(x, verbose=0)
        predicted_class = np.argmax(predictions)

        return self.convert_class_to_emotion(predicted_class)
