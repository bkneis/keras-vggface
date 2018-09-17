import pickle
import sys
from argparse import ArgumentParser

import numpy as np
import keras
import cv2
from keras.models import load_model
from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace

from dataset import get_training_data


def parse_arguments(args):
    parser = ArgumentParser()
    parser.add_argument('--disable_training_cache', action='store_true')
    parser.add_argument('--model', type=str, default=None)

    return parser.parse_args(args)


def main(args):
    """
    Application to train a neural network through transfer learning of 2D images to 3D surface properties
    """

    # Get the training data, corresponding labels and number of classes
    try:
        # Check if we should specifically not use the training data cache
        if args.disable_training_cache:
            raise FileNotFoundError

        # Try and load the cached training data
        print('Attempting to load training data from cache')
        data, labels, nb_class = pickle.load(open("training_cache.h5", "rb"))
        print('Training data loaded from cache')

    except FileNotFoundError:
        # If not found compute the training data
        print('Training data not cached, preparing the data now')
        data, labels, nb_class = get_training_data()
        pickle.dump((data, labels, nb_class), open("training_cache.h5", "wb"))
        print("Training data prepared and cached to 'training_cache.h5'")

    # Convert labels to binary vectors
    labels = keras.utils.to_categorical(labels, nb_class)

    # Check is we should create a new model or use a saved one
    if args.model is None:
        # Create the VGGFace model
        vgg_model = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))

        # Recreate the classification layers
        last_layer = vgg_model.get_layer('avg_pool').output
        x = Flatten(name='flatten')(last_layer)
        out = Dense(nb_class, activation='softmax', name='classifier')(x)

        # Create new model using VGGFace as input with new classification layers
        custom_vgg_model = Model(vgg_model.input, out)
        custom_vgg_model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])

        # Train our model
        custom_vgg_model.fit(data, labels, epochs=100, batch_size=32)
        custom_vgg_model.save('resnet50.h5')

    else:
        # Load the model provided through args
        try:
            custom_vgg_model = load_model(args.model)
        except FileNotFoundError:
            print('Model provided via --model could not ben found')
            return

    # Test the model
    test_img = cv2.imread('../image/test2.png')
    test_img = cv2.resize(test_img, (224, 224))
    pred = custom_vgg_model.predict(np.array([test_img]))
    print('prediction is ', pred)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[-1:]))
