"""
This module is MLP implementation on MNIST data
    with TF 2+ Using Keras Functional API

* More flexible than sequential api
* Allows to create multiple input and output models
* Allows to define AD-HOC Acyclic Network graph

* Models are defined as instances of layers
    - Create Standalone input layer
    - Create Standalone hidden layers
    - Create new layers with operations like concatenate, etc
    - Create Standalone output layer
"""

import os
import numpy as np
import matplotlib.pyplot as plt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.utils import to_categorical


class MLP:

    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.model_path = f'{os.path.basename(os.path.realpath(__file__))}.h5'
        self.image_size = self.x_train.shape[1]
        self.input_size = self.image_size * self.image_size
        self.pre_process_data()
        self.model = self.create_model()
        self.batch_size = 128

    def view_data(self):
        # view dataset
        image_indexes = np.random.randint(0, self.x_train.shape[0], size=3)
        train_images = self.x_train[image_indexes]
        plt.figure(figsize=(5, 5))
        for index in range(len(image_indexes)):
            plt.subplot(5, 5, index + 1)
            image = train_images[index]
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.show()

    def pre_process_data(self):
        # image pre-processing
        self.x_train = np.reshape(self.x_train, [-1, self.input_size])
        self.x_train = self.x_train.astype('float32') / 255
        self.x_test = np.reshape(self.x_test, [-1, self.input_size])
        self.x_test = self.x_test.astype('float32') / 255

    def create_model(self):
        num_of_labels = len(np.unique(self.y_train))

        # labels to categorical
        self.y_test = to_categorical(self.y_test)
        self.y_train = to_categorical(self.y_train)

        # network params
        hidden_units = 256
        dropout = 0.3

        # define model

        # define input layer
        input_layer = keras.layers.Input(shape=(self.input_size, ), name='input_layer')

        # H1 Layer
        h1_layer = keras.layers.Dense(hidden_units, activation=keras.activations.relu)(input_layer)
        dropout_layer_1 = keras.layers.Dropout(dropout)(h1_layer)

        h2_layer = keras.layers.Dense(hidden_units, activation=keras.activations.relu)(input_layer)
        dropout_layer_2 = keras.layers.Dropout(dropout)(h2_layer)

        # merge
        merge = keras.layers.concatenate([dropout_layer_1, dropout_layer_2])

        # Output Layer
        output_layer = keras.layers.Dense(num_of_labels, activation=keras.activations.softmax)(merge)

        # create model
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        print(model.summary())

        return model

    def train(self):
        # define loss function
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adam(),
                           metrics=['accuracy'])

        callback = [keras.callbacks.EarlyStopping(patience=2, monitor='val_loss')
                    ]
        self.model.fit(self.x_train, self.y_train, epochs=30, batch_size=self.batch_size,
                       callbacks=callback)

        # validate model
        _, acc = self.model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size,
                                     verbose=0)
        print(f'Test Accuracy {100.0 * acc: .2f}')

        self.model.save(self.model_path)

    def load_model(self):
        loaded_model = keras.models.load_model(self.model_path)
        _, acc = loaded_model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size,
                                       verbose=0)
        print(f'Loaded Model Test Accuracy {100.0 * acc: .2f}')


if __name__ == '__main__':
    mlp_digit = MLP()
    mlp_digit.train()
    mlp_digit.load_model()
