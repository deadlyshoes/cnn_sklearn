import keras
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from keras.datasets import mnist
from keras.utils import np_utils

from sklearn.metrics import accuracy_score

import numpy as np

class Keras_CNN_estimator():
    model = None

    def __init__(self, kernel_size_1=0.1, kernel_size_2=0.1, activation_1=0, activation_2=0, 
                pooling_1=0, pooling_2=0, n_kernels_1=0.001, n_kernels_2=0.001, neurons_1=0.01, neurons_2=0.01, 
                activation_full_1=0, activation_full_2=0, learning_rate=0.1, dropout_1=0.1, dropout_2=0.3):

        self.kernel_size_1 = kernel_size_1
        self.kernel_size_2 = kernel_size_2
        self.activation_1 = activation_1
        self.activation_2 = activation_2
        self.neurons_1 = neurons_1
        self.neurons_2 = neurons_2
        self.pooling_1 = pooling_1
        self.pooling_2 = pooling_2
        self.activation_full_1 = activation_full_1
        self.activation_full_2 = activation_full_2
        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2
        self.learning_rate = learning_rate
        self.n_kernels_1 = n_kernels_1
        self.n_kernels_2 = n_kernels_2

        self.build_cnn()

    def get_params(self, deep=True):
        return {
            "kernel_size_1": self.kernel_size_1,
            "kernel_size_2": self.kernel_size_2,
            "activation_1": self.activation_1,
            "activation_2": self.activation_2,
            "neurons_1": self.neurons_1,
            "neurons_2": self.neurons_2,
            "pooling_1": self.pooling_1,
            "pooling_2": self.pooling_2,
            "activation_full_1": self.activation_full_1,
            "activation_full_2": self.activation_full_2,
            "dropout_1": self.dropout_1,
            "dropout_2": self.dropout_2,
            "learning_rate": self.learning_rate,
            "n_kernels_1": self.n_kernels_1,
            "n_kernels_2": self.n_kernels_2
        }

    def set_params(self, parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        self.build_cnn()

        return self

    def fit(self, X, y):
        history = model.fit(X, y, epochs=30, batch_size=200, verbose=1)

        return self

    def predict(self, X):
        y_pred = model.predict(X)
        return y_pred

    def build_cnn(self):
        activations = ['relu', 'sigmoid', 'tanh']
        poolings = ['max-pooling', 'average-pooling']
        optimizers = ['adam', 'sgd']
        initializers = ['random_uniform', 'normal']

        kernel_size_1 = int(round(self.kernel_size_1 * 10)) * 2 + 1
        kernel_size_2 = int(round(self.kernel_size_2 * 10)) * 2 + 1
        activation_1 = activations[int(round(self.activation_1 * 10))]
        activation_2 = activations[int(round(self.activation_2 * 10))]
        pooling_1 = poolings[int(round(self.pooling_1 * 10))]
        pooling_2 = poolings[int(round(self.pooling_2 * 10))]
        n_kernels_1 = int(self.n_kernels_1 * 1000)
        n_kernels_2 = int(self.n_kernels_2 * 1000)
        neurons_1 = int(self.neurons_1 * 1000)
        neurons_2 = int(self.neurons_2 * 1000)
        activation_full_1 = activations[int(self.activation_full_1 * 10)]
        activation_full_2 = activations[int(self.activation_full_2 * 10)]
        learning_rate = self.learning_rate
        dropout_1 = self.dropout_1
        dropout_2 = self.dropout_2

        model = Sequential()
        model.add(Conv2D(n_kernels_1, kernel_size=kernel_size_1, activation=activation_1))
        if pooling_1 == 'max-pooling':
            model.add(MaxPooling2D())
        else:
            model.add(AveragePooling2D())
        model.add(Conv2D(n_kernels_2, kernel_size=kernel_size_2, activation=activation_2))
        if pooling_2 == 'max-pooling':
            model.add(MaxPooling2D())
        else:
            model.add(AveragePooling2D())
        model.add(Flatten())
        model.add(Dense(neurons_1, activation=activation_full_1))
        model.add(Dropout(dropout_1))
        model.add(Dense(neurons_2, activation=activation_full_2))
        model.add(Dropout(dropout_2))


    
    
    
