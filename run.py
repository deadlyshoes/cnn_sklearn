import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.svm import SVC,SVR
from sklearn import datasets
import scipy.stats as stats

from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils

from estimator import Keras_CNN_estimator

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1)
X_train = X_train.astype('float32')
X_train /= 255

neurons_values = list(range(10, 800))
for i in range(len(neurons_values)):
    neurons_values[i] /= 1000.0

n_kernels_values = list(range(1, 100))
for i in range(len(n_kernels_values)):
    n_kernels_values[i] /= 1000.0

# Random Search
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
rf_params = {
    "kernel_size_1": [0.1, 0.2, 0.3, 0.4],
    "kernel_size_2": [0.1, 0.2, 0.3, 0.4],
    "activation_1": [0, 0.1, 0.2],
    "activation_2": [0, 0.1, 0.2],
    "neurons_1": neurons_values,
    "neurons_2": neurons_values,
    "pooling_1": [0, 0.1],
    "pooling_2": [0, 0.1],
    "activation_full_1": [0, 0.1, 0.2],
    "activation_full_2": [0, 0.1, 0.2],
    "dropout_1": stats.uniform(0.1, 0.9),
    "dropout_2": stats.uniform(0.1, 0.9),
    "learning_rate": stats.uniform(0.001, 1),
    "n_kernels_1": n_kernels_values,
    "n_kernels_2": n_kernels_values
}
n_iter_search=100
clf = Keras_CNN_estimator()
Random = RandomizedSearchCV(clf, param_distributions=rf_params,n_iter=n_iter_search,cv=3,scoring='accuracy')
Random.fit(X_train, y_train)
print(Random.best_params_)
print("Accuracy:"+ str(Random.best_score_))

