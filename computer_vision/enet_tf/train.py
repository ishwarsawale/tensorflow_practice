from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import tensorflow as tf
import matplotlib.pylab as plt
import numpy as np

from utils import preprocess_img_label, map_singlehead, map_doublehead, map_label, tf_dataset_generator, get_class_weights
from models import EnetModel

parser = argparse.ArgumentParser()
parser.add_argument("--data_location", type=str, default='/home/robot/Downloads/camdata/dataset')
args = parser.parse_args()


# creating datasets
base_path = args.data_location
img_pattern = f"{base_path}/train/images/*.png"
label_pattern = f"{base_path}/train/labels/*.png"
img_pattern_val = f"{base_path}/val/images/*.png"
label_pattern_val = f"{base_path}/val/labels/*.png"
img_pattern_test = f"{base_path}/test/images/*.png"
label_pattern_test = f"{base_path}/test/labels/*.png"

# batch size
batch_size = 8

# image size
img_height = 360
img_width = 480
h_enc = img_height // 8
w_enc = img_width // 8
h_dec = img_height
w_dec = img_width

# create (img,label) string tensor lists
filelist_train = preprocess_img_label(img_pattern, label_pattern)
filelist_val = preprocess_img_label(img_pattern_val, label_pattern_val)
filelist_test = preprocess_img_label(img_pattern_test, label_pattern_test)

# training dataset size
n_train = tf.data.experimental.cardinality(filelist_train).numpy()
n_val = tf.data.experimental.cardinality(filelist_val).numpy()
n_test = tf.data.experimental.cardinality(filelist_test).numpy()

# define mapping functions for single and double head nets
map_single = lambda img_file, label_file: map_singlehead(
    img_file, label_file, h_dec, w_dec)
map_double = lambda img_file, label_file: map_doublehead(
    img_file, label_file, h_enc, w_enc, h_dec, w_dec)

# create single head datasets
train_single_ds = filelist_train.shuffle(n_train).map(map_single).cache().batch(batch_size).repeat()
val_single_ds = filelist_val.map(map_single).cache().batch(batch_size).repeat()
test_single_ds = filelist_test.map(map_single).cache().batch(batch_size).repeat()

# create double head datasets
train_double_ds = filelist_train.shuffle(n_train).map(map_double).cache().batch(batch_size).repeat()
val_double_ds = filelist_val.map(map_double).cache().batch(batch_size).repeat()
test_double_ds = filelist_test.map(map_double).cache().batch(batch_size).repeat()

# get class weights
label_filelist = tf.data.Dataset.list_files(label_pattern, shuffle=False)
label_ds = label_filelist.map(lambda x: map_label(x, h_dec, w_dec))
class_weights = get_class_weights(label_ds).tolist()


def train_enet():

    Enet = EnetModel(C=12,MultiObjective=True,l2=1e-3)
    # Train Encoder
    for layer in Enet.layers[-6:]:
      layer.trainable = False

    n_epochs = 60
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
    Enet.compile(optimizer=adam_optimizer,
                 loss=['sparse_categorical_crossentropy','sparse_categorical_crossentropy'],
                 metrics=['accuracy','accuracy'],
                 loss_weights=[1.0,0.0])

    enet_enc_history = Enet.fit(x= train_double_ds,
            epochs=n_epochs,
            steps_per_epoch=n_train//batch_size,
            validation_data= val_double_ds,
            validation_steps=n_val//batch_size//5,
            class_weight=[class_weights,class_weights])

    # Train Decoder
    for layer in Enet.layers[-6:]:
      layer.trainable = True
    for layer in Enet.layers[:-6]:
      layer.trainable = False

    n_epochs = 60
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
    Enet.compile(optimizer=adam_optimizer,
                 loss=['sparse_categorical_crossentropy','sparse_categorical_crossentropy'],
                 metrics=['accuracy','accuracy'],
                 loss_weights=[0.0,1.0])

    enet_dec_history = Enet.fit(x= train_double_ds,
            epochs=n_epochs,
            steps_per_epoch=n_train//batch_size,
            validation_data= val_double_ds,
            validation_steps=n_val//batch_size//5,
            class_weight=[class_weights,class_weights])

    Enet.evaluate(x=test_double_ds,steps=n_test//batch_size)

    loss = enet_dec_history.history['loss']
    val_loss = enet_dec_history.history['val_loss']
    acc = enet_dec_history.history['output_2_accuracy']
    val_acc = enet_dec_history.history['val_output_2_accuracy']

    Enet.save_weights('./weights/Enet.tf')
    epochs = range(n_epochs)

    plt.figure(figsize=(12,8))
    plt.plot(epochs, loss/np.max(loss), 'r', label='Training loss')
    plt.plot(epochs, val_loss/np.max(val_loss), 'b', label='Validation loss')
    plt.plot(epochs, acc, 'r:', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b:', label='Validation accuracy')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train_enet()
