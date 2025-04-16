import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, Input, Flatten, Dense, MaxPooling2D


## function plotting accuracy and loss of training vs validation dataset
def plot_accuracy_loss(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss=history.history['loss']
    val_loss=history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(14, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

## function building the model used in this task
def build_model(img_height, img_width, metrics):
    ResNet_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights="imagenet",
    )

    ResNet_model.trainable = False
    inputs = Input(shape=(img_height, img_width, 3))
    x = ResNet_model(inputs, training = False)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(32, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation = 'sigmoid')(x)
    model = tf.keras.Model(
        inputs,
        outputs
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics= metrics)

    return model
