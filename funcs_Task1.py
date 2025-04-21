import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Distribution of classes in the data set
def get_distribution_barplot(data, set):
    class_labels = data.class_indices
    labels = data.labels
    class_counts = {label: sum(labels == i) for i, label in enumerate(class_labels)}
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xlabel('Class')
    plt.ylabel('Number of samples')
    plt.title('Class Distribution in {set} Set'.format(set = set))
    plt.show()

# plot 5 images with label
def plotImages(images_arr, labels):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for i in range(len(axes)):
        axes[i].imshow(images_arr[i])
        axes[i].set_title("COVID" if labels[i] == 1 else "Normal")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

# Function to generate augmented images
def get_augmented_images(datagen, train_dir, batch_size=32, img_size=(128, 128), num_images=5):
    train_data_gen = datagen.flow_from_directory(
        batch_size=batch_size,
        directory=train_dir,
        shuffle=True,
        target_size=img_size
    )
    augmented_images = [train_data_gen[0][0][0] for _ in range(num_images)]
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for img, ax in zip(augmented_images , axes):
        ax.imshow(img)
        ax.axis('off')
    plt.show()
    return [train_data_gen[0][0][0] for _ in range(num_images)]  

#################################
### Define pipeline functions ###
#################################
def get_normalization_numbers(img_size, train_dir):
    # get parameters for normalization
    train_image_generator_1 = ImageDataGenerator(rescale=1./255)
    train_data_gen_1 = train_image_generator_1.flow_from_directory(
        batch_size=1600,
        directory=train_dir,
        shuffle=True,
        target_size=img_size,
        class_mode = "binary"
    )
    sample_batch, labels = next(train_data_gen_1)
    global_mean = np.mean(sample_batch)
    global_std = np.std(sample_batch)

    return global_mean, global_std

def training_data_Pipeline(batch_size, img_size, train_dir):
    # get parameters for normalization
    global_mean, global_std = get_normalization_numbers(img_size, train_dir)

    #set up training data generator, with data augmentation and normalization
    train_image_generator = ImageDataGenerator(
        rescale=1./255, 
        featurewise_center=True, 
        featurewise_std_normalization=True,
        rotation_range = 15,
        zoom_range = 0.2,
        width_shift_range=0.2
    )
    
    train_image_generator.mean = global_mean
    train_image_generator.std = global_std  

    train_data_gen = train_image_generator.flow_from_directory(
        batch_size=batch_size,
        directory=train_dir,
        shuffle=True,
        target_size=img_size,
        class_mode = "binary"
    )

    return(train_data_gen)


def test_validation_data_Pipeline(batch_size, img_size, train_dir, test_dir, shuffle = True):
    # get parameters for normalization
    global_mean, global_std = get_normalization_numbers(img_size, train_dir)

    #set up training data generator, with data augmentation and normalization
    test_image_generator = ImageDataGenerator(
        rescale=1./255, 
        featurewise_center=True, 
        featurewise_std_normalization=True
    )
    
    test_image_generator.mean = global_mean
    test_image_generator.std = global_std  

    test_data_gen = test_image_generator.flow_from_directory(
        batch_size=batch_size,
        directory=test_dir,
        shuffle=shuffle,
        seed = 123,
        target_size=img_size,
        class_mode = "binary"
    )

    return(test_data_gen)

def test_data_Pipeline_non_normalized(batch_size, img_size, test_dir):
    #set up training data generator, with data augmentation and normalization
    test_image_generator = ImageDataGenerator(
        rescale=1./255, 
    )
    
    test_data_gen = test_image_generator.flow_from_directory(
        batch_size=batch_size,
        directory=test_dir,
        shuffle=False,
        seed = 123,
        target_size=img_size,
        class_mode = "binary"
    )

    return(test_data_gen)