from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib.pylab as plt

from utils import preprocess_img_label, map_singlehead, map_doublehead
from models import EnetModel

# creating datasets
base_path = '/home/robot/Downloads/camdata/dataset'
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

train_single_ds = filelist_train.shuffle(n_train).map(map_single).cache().batch(batch_size).repeat()
val_single_ds = filelist_val.map(map_single).cache().batch(batch_size).repeat()
test_single_ds = filelist_test.map(map_single).cache().batch(batch_size).repeat()

Enet = EnetModel(C=12, MultiObjective=True, l2=1e-3)
Enet.load_weights('./weights/Enet.tf')


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


cnt = 0
for img, iml in train_single_ds.take(5):
    img_test = img
    iml_test = iml
    img_enc_probs, img_dec_probs = Enet(img_test[0:1, :, :, :])
    img_dec_out = create_mask(img_dec_probs)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1)
    plt.xticks([])
    plt.yticks([])
    plt.title('Image', fontdict={'fontsize': 20})
    plt.imshow(img_test.numpy()[0, :, :, :])

    plt.subplot(1, 3, 2)
    plt.xticks([])
    plt.yticks([])
    plt.title('Ground Truth', fontdict={'fontsize': 20})
    plt.imshow(iml_test.numpy()[0, :, :, 0])

    plt.subplot(1, 3, 3)
    plt.imshow(img_dec_out[:, :, 0])
    plt.xticks([])
    plt.yticks([])
    plt.title('Generated', fontdict={'fontsize': 20})

    plt.tight_layout()
    plt.savefig(f'./output_{cnt}.png')
    cnt += 1
    plt.show()
