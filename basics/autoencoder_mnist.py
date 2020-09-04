from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

import numpy as np
import matplotlib.pyplot as plt


def prepare_dataset():
    # load dataset
    (x_train, _), (x_test, _) = mnist.load_data()
    print(f'shape of input x_train is {x_train.shape}')
    # reshape dataset to (num_examples, 28, 28, 1)
    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    # normalize image
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    return x_train, x_test


def create_network(x_train, x_test):
    # network params
    image_size = x_train.shape[1]
    input_shape = (image_size, image_size, 1)
    batch_size = 32
    kernel_size = 3
    latent_dim = 16

    # CNN filters
    layer_filters = [32, 64]

    # encoder
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs  # input shape is 28, 28, 1
    """
    Conv shape calculation
    w = input shape
    K = kernel size
    P = padding
    S = Stride
    
    [W-K+2P)/S]+1
    
    For input 28, K = 3, P=0, S = 2
    x1 = [(28-3+2*0)/2] +1 = 14 where filter size = 32
    (x1 output shape = 14, 14, 32)
    x2 = [(14-3+2*0)/2]+1 = 7 where filter size = 64
    (x2 output shape = 7, 7, 64)
    """
    for filters in layer_filters:
        x = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', strides=2, padding='same')(x)

    # encoder output is 7,7,64
    shape = K.int_shape(x)  # get shape of encoder

    # create latent vector
    x = Flatten()(x)
    latent = Dense(latent_dim, name='latent_vector')(x)

    # create encoder model

    encoder = Model(inputs, latent, name='encoder')
    encoder.summary()
    plot_model(encoder, to_file='encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    for filters in layer_filters[::-1]:
        x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, activation='relu',
                            strides=2, padding='same')(x)

    outputs = Conv2DTranspose(filters=1, kernel_size=kernel_size, activation='sigmoid',
                              padding='same', name='decoder_output')(x)

    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    plot_model(decoder, to_file='decoder.png', show_shapes=True)

    autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
    autoencoder.summary()
    plot_model(autoencoder, to_file='autoencoder.png', show_shapes=True)

    autoencoder.compile(loss='mse', optimizer='adam')

    autoencoder.fit(x_train, x_train, validation_data=(x_test, x_test), epochs=1,
                    batch_size=batch_size)
    x_decoded = autoencoder.predict(x_test)

    imgs = np.concatenate([x_test[:8], x_decoded[:8]])
    imgs = imgs.reshape((4, 4, image_size, image_size))
    imgs = np.vstack([np.hstack(i) for i in imgs])
    plt.figure()
    plt.axis('off')
    plt.title('Input row 0, 1 & Output row 2, 3')
    plt.imshow(imgs, interpolation='none', cmap='gray')
    plt.savefig('input_and_decoded.png')
    plt.show()


if __name__ == '__main__':
    x_train, x_test = prepare_dataset()
    create_network(x_train, x_test)
