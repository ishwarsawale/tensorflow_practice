import tensorflow as tf
import argparse
import os

from resnet import resnet_18, resnet_34, resnet_50, resnet_101, resnet_152


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='resnet18')
parser.add_argument("--image_size", type=int, default=180)
parser.add_argument("--channels", type=int, default=3)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--save_dir", type=str, default='graph')
parser.add_argument("--dataset_dir", type=str, default='training_data')

args = parser.parse_args()


def get_model():
    model = resnet_50(args.num_classes)
    if args.model == "resnet18":
        model = resnet_18(args.num_classes)
    if args.model == "resnet34":
        model = resnet_34(args.num_classes)
    if args.model == "resnet101":
        model = resnet_101(args.num_classes)
    if args.model == "resnet152":
        model = resnet_152(args.num_classes)
    model.build(input_shape=(None, args.image_size, args.image_size, args.channels))
    model.summary() return model


def train():
    """
    Training sample for Cat & Dog dataset
    https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip
    :return:
    """
    # delete bad images
    num_skipped = 0
    for folder_name in ('Cat', 'Dog'):
        folder_path = os.path.join(args.dataset_dir, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            fobj = open(fpath, 'rb')
            if tf.compat.as_bytes('JFIF') not in fobj.peek(10):
                num_skipped += 1
                # Delete corrupted image
                os.system('rm ' + fpath)
    print('Deleted %d images' % num_skipped)

    # data pre-processing
    batch_size = args.batch_size
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        'input_dataset', validation_split=0.2, subset='training', seed=1337,
        image_size=(180, 180), batch_size=batch_size)
    valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        'input_dataset', validation_split=0.2, subset='validation', seed=1337,
        image_size=(180,180), batch_size=batch_size)

    # create model
    model = get_model()

    # define loss and optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adadelta()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(y_true=labels, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def valid_step(images, labels):
        predictions = model(images, training=False)
        v_loss = loss_object(labels, predictions)

        valid_loss(v_loss)
        valid_accuracy(labels, predictions)

    # start training
    for epoch in range(args.epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()
        step = 0
        for images, labels in train_dataset:
            step += 1
            train_step(images, labels)

            print(f'Epoch: {epoch} step: {step} loss: {train_loss.result()}'
                  f'accuracy: {train_accuracy.result()}')

        for valid_images, valid_labels in valid_dataset:
            valid_step(valid_images, valid_labels)

        print(f"Epoch: {epoch} train loss: {train_loss.result()} train accuracy: {train_accuracy.result()}"
              f"valid loss: {valid_loss.result()} valid accuracy: {valid_accuracy.result()}")

    model.save_weights(filepath=args.save_dir, save_format='tf')


if __name__ == '__main__':

    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    train()
