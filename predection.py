import keras
import librosa
import numpy as np
from keras.layers import Input, Conv1D, Flatten, Dense, Dropout, Activation, MaxPooling1D
from keras.models import Model
from flask import Flask, request, redirect, render_template_string, send_from_directory
from werkzeug.utils import secure_filename
import os
from io import BytesIO
import base64
import os

''' Class to Make Predicitions '''

class LivePredictions:
    def __init__(self):
        self.path = r"C://Users//kstar//Documents//Projects//Final Year Project//One More Trial//model//Emotion_Voice_Detection_Model.h5"

        input_layer = Input(shape=(40, 1))
        x = Conv1D(64, 5, padding='same')(input_layer)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Dense(8)(x)
        output_layer = Activation('softmax')(x)
        self.model = Model(inputs=input_layer, outputs=output_layer)
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
    
if __name__ == "__main__":
    pred = LivePredictions()
    pred.model.summary()

    # wrong_classification_path = '/content/03-01-01-01-01-02-05.wav'
    correct_classification_path = 'C://Users//kstar//Documents//Projects//Final Year Project//One More Trial//examples//10-16-07-29-82-30-63.wav'

    correct = pred.make_predictions(file=correct_classification_path)
    print(f"Correct Emotion Is {correct}")
    # wrong = predection.make_predictions(file=wrong_classification_path)
    # print(f"Wrong Emotion Is {wrong}")