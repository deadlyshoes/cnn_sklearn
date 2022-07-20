class Keras_CNN_estimator():
    def __init__(self, kernel_size_1=3, kernel_size_2=3, activation_1="relu", activation_2="relu", 
                pooling_1="max-pooling", pooling_2="max-pooling", n_kernels_1=150, n_kernels_2=75, neurons_1=50, neurons_2=50, 
                activation_full_1="sigmoid", activation_full_2="sigmoid", learning_rate=0.1, dropout_1=0.1, dropout_2=0.3):
        lbs = [0.0 for _ in range(15)]
        ups = [1.0 for _ in range(15)]

        # kernel size 1
        lbs[0] = 0.1
        ups[0] = 0.4

        # kernel size 2
        lbs[1] = 0.1
        ups[1] = 0.4

        # activation 1
        lbs[2] = 0
        ups[2] = 0.2

        # activation 2
        lbs[3] = 0
        ups[3] = 0.2

        # kind pooling 1
        lbs[4] = 0
        ups[4] = 0.1

        # kind pooling 2
        lbs[5] = 0
        ups[5] = 0.1

        # n kernels 1
        lbs[6] = 0.001
        ups[6] = 0.1       

        # n kernels 2
        lbs[7] = 0.001
        ups[7] = 0.1

        # neurons 1
        lbs[8] = 0.01
        ups[8] = 0.8

        # neurons 2
        lbs[9] = 0.01
        ups[9] = 0.8

        # activation full 1
        lbs[10] = 0
        ups[10] = 0.2

        # activation full 2
        lbs[11] = 0
        ups[11] = 0.2

        # learning rate
        lbs[12] = 0.001
        ups[12] = 1

        # dropout rate 1
        lbs[13] = 0.1
        ups[13] = 0.9

        # dropout rate 2
        lbs[14] = 0.1
        ups[14] = 0.9

        self.lowerBounds = lbs
        self.upperBounds = ups

        self.activations = ['relu', 'sigmoid', 'tanh']
        self.poolings = ['max-pooling', 'average-pooling']
        self.optimizers = ['adam', 'sgd']
        self.initializers = ['random_uniform', 'normal']

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
            "learning rate": self.learning_rate,
            "n_kernels_1": self.n_kernels_1,
            "n_kernels_2": self.n_kernels_2
        }

    def set_params(self, parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        self.build_cnn()

        return self

    def fit(self, X, y):
        history = self.model.fit(self.X_train, self.y_train, epochs=30, batch_size=200, verbose=1)

        return self

    def predict(self, X):
        y_pred = model.predict(self.X)
        return y_pred

    def build_cnn(self):
        kernel_size_1 = int(round(self.kernel_size_1 * 10)) * 2 + 1
        kernel_size_2 = int(round(self.kernel_size_2 * 10)) * 2 + 1
        activation_1 = self.activations[int(round(self.activation_1 * 10))]
        activation_2 = self.activations[int(round(self.activation_2 * 10))]
        pooling_1 = self.poolings[int(round(self.pooling_1 * 10))]
        pooling_2 = self.poolings[int(round(self.pooling_2 * 10))]
        n_kernels_1 = int(self.n_kernels_1 * 1000)
        n_kernels_2 = int(self.n_kernels_2 * 1000)
        neurons_1 = int(self.neurons_1 * 1000)
        neurons_2 = int(self.neurons_2 * 1000)
        activation_full_1 = self.activations[int(self.activation_full_1 * 10)]
        activation_full_2 = self.activations[int(self.activation_full_2 * 10)]
        learning_rate = self.learning_rate
        dropout_1 = self.dropout_1
        dropout_2 = self.dropout_2

        model = Sequential()
        model.add(Conv2D(n_kernels_1, kernel_size=kernel_size_1, activation=activation_1, input_shape=self.input_dim))
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

        self.model = model


    
    
    
