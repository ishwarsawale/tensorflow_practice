"""
This module is MLP implementation on MNIST data
    with TF 2+ Using Keras Subclassing API
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.utils import to_categorical


class MyModel(keras.Model):

  def __init__(self, num_classes=10, input_shape=784):
    super(MyModel, self).__init__(name='my_model')
    self.num_classes = num_classes
    # Define your layers here.
    self.dense_1 = keras.layers.Dense(32, activation='relu')
    self.dense_2 = keras.layers.Dense(num_classes, activation='sigmoid')

  def call(self, inputs):
    # Define your forward pass here,
    # using layers you previously defined (in `__init__`).
    x = self.dense_1(inputs)
    return self.dense_2(x)

  def compute_output_shape(self, input_shape):
    # You need to override this function if you want to use the subclassed model
    # as part of a functional-style model.
    # Otherwise, this method is optional.
    shape = tf.TensorShape(input_shape).as_list()
    shape[-1] = self.num_classes
    return tf.TensorShape(shape)


class MLP():

    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.model_path = f'{os.path.basename(os.path.realpath(__file__))}'
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

        # Instantiates the subclassed model.
        model = MyModel(num_classes=10)

        model.build(input_shape=(None, 784))
        # # define loss function
        model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adam(),
                           metrics=['accuracy'])
        print(model.summary())
        return model

    def train(self):

        callback = [keras.callbacks.EarlyStopping(patience=2, monitor='val_loss')
                    ]
        self.model.fit(self.x_train, self.y_train, epochs=20, batch_size=self.batch_size,
                       callbacks=callback)

        # validate model
        _, acc = self.model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size,
                                     verbose=0)
        print(f'Test Accuracy {100.0 * acc: .2f}')

        self.model.save_weights('test.h5f', save_format='tf')

    def load_model(self):
        new_model = self.create_model()
        loaded_model = new_model.load_weights('test.h5f')
        y_test_bar = loaded_model.predict(self.x_test)
        acc = accuracy_score(self.y_test, y_test_bar)

        print(f'Loaded Test Accuracy {100.0 * acc: .2f}')


if __name__ == '__main__':
    mlp_digit = MLP()
    mlp_digit.train()
    mlp_digit.load_model()
