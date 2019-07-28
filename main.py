import tensorflow as tf
import os, signal
import argparse
from tensorflow._api.v1.keras.optimizers import RMSprop
from tensorflow._api.v1.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path


def free_memory():
    os.kill(os.getpid(),
            signal.SIGKILL)
    print('Freeing memory resources')


def main(gpu, epochs):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    home = str(Path.home())
    base_dir = home + '/.keras/datasets/cats_and_dogs_filtered'
    img_size = 224
    batch_size = 20
    seed = 21

    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    classes_dictionary = dict()
    for num, name in enumerate(os.listdir(train_dir)):
        classes_dictionary[num] = name

    train_class0_dir = os.path.join(train_dir, classes_dictionary[0])
    train_class1_dir = os.path.join(train_dir, classes_dictionary[1])
    validation_class0_dir = os.path.join(validation_dir, classes_dictionary[0])
    validation_class1_dir = os.path.join(validation_dir, classes_dictionary[1])

    # printing overall number of images
    print('Total train for class {} is {}'.format(classes_dictionary[0], len(os.listdir(train_class0_dir))))
    print('Total train for class {} is {}'.format(classes_dictionary[1], len(os.listdir(train_class1_dir))))
    print('Total validation for class {} is {}'.format(classes_dictionary[0], len(os.listdir(validation_class0_dir))))
    print('Total validation for class {} is {}'.format(classes_dictionary[1], len(os.listdir(validation_class1_dir))))

    train_data_gen = ImageDataGenerator(rescale=1. / 255,
                                        rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True)

    validation_data_gen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_data_gen.flow_from_directory(directory=train_dir,
                                                         target_size=(img_size, img_size),
                                                         batch_size=batch_size,
                                                         class_mode='binary',
                                                         seed=seed,
                                                         # color_mode='grayscale',
                                                         save_to_dir=None)

    validation_generator = validation_data_gen.flow_from_directory(directory=validation_dir,
                                                                   target_size=(img_size, img_size),
                                                                   batch_size=batch_size,
                                                                   seed=seed,
                                                                   # color_mode='grayscale',
                                                                   class_mode='binary')

    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 150x150 with 3 bytes color
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.summary()
    model.compile(optimizer=RMSprop(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    history = model.fit_generator(train_generator,
                                  validation_data=validation_generator,
                                  steps_per_epoch=100,
                                  epochs=epochs,
                                  validation_steps=50,
                                  verbose=2)

    validation_steps = validation_generator.n // batch_size
    loss, acc = model.evaluate_generator(generator=validation_generator,
                                         steps=validation_steps,
                                         verbose=1)

    print('trained model loss {:.4f}, trained model accuracy {:.4f}'.format(loss, acc))

    free_memory()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for retraining ConvNets')
    parser.add_argument(
        "--gpu",
        default="0",
        type=str,
        help="Number of the gpu to use for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of the gpu to use for training"
    )
    args = parser.parse_args()
    main(gpu=args.gpu)
