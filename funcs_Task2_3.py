import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, Input, Flatten, Dense, MaxPooling2D
from sklearn.metrics import confusion_matrix
import seaborn as sns

##########################
### plotting functions ###
##########################

## function plotting accuracy and loss of training vs validation dataset
def plot_accuracy_loss(history, epochs, validation = True):
    acc = history.history['accuracy']
    loss=history.history['loss']
    
    if validation:
        val_acc = history.history['val_accuracy']
        val_loss=history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    if validation:
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    if validation:
        plt.title('Training and Validation Accuracy')
    else:
        plt.title("Training Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    if validation:
        plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if validation:
        plt.title('Training and Validation Loss')
    else:
        plt.title("Training Loss")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, test_data_gen):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_data_gen.class_indices.keys(), yticklabels=test_data_gen.class_indices.keys())
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

#########################
### utility functions ###
#########################

## function making it possible to concatenate validation and training data generators
def concat_generators(*gens):
    for gen in gens:
        yield from gen
