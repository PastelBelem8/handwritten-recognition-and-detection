# ------------------------------------------------- #
#                   Imports                         #
# ------------------------------------------------- #
# Graphical Library
import matplotlib.pyplot as plt 

# Image loading
import cv2 

# Maths
import math
import numpy as np
from scipy import ndimage

# Model Persistence
import pickle

# Iterable utils
import itertools

import keras
from keras.utils import np_utils

# ------------------------------------------------- #
#                Global constants                   #
# ------------------------------------------------- #
model_filename = "trained_mlp.sav"

number_of_writers = range(1, 5)
number_of_digits = range(0, 10)
img_options = ['no_flash', 'flash']

# The samples of the test set that we pick for analyzing
selected_examples = [0, 9, 53, 30]

# ------------------------------------------------- #
#                   Setup                           #
# ------------------------------------------------- #
def get_img_filepath(writer, digit, img_option, folder='imgs', extension="jpg"):
    return f"{folder}/{img_option}/all_data/{writer}_{digit}.{extension}"


# Sanity check
print(get_img_filepath(2, 1, 'no_flash'))
print(get_img_filepath(2, 1, 'flash'))


def get_test_set_color(writers, digits, img_options):
    X_test, y_test = [], []
    for i in itertools.product(writers, digits, img_options):
        img_filepath = get_img_filepath(*i)

        X_test += [cv2.imread(img_filepath)]
        y_test += [i[1]]
        
    return np.array(X_test), np.array(y_test)

def get_test_set_bw(writers, digits, img_options):
    X_test, y_test = [], []
    for i in itertools.product(writers, digits, img_options):
        img_filepath = get_img_filepath(*i)

        X_test += [cv2.imread(img_filepath, cv2.IMREAD_GRAYSCALE)]
        y_test += [i[1]]
        
    return np.array(X_test), np.array(y_test)

# Sanity check
print("Color images dimensions:", get_test_set_color(range(1, 2), [1, 2, 3], ['no_flash'])[0][0].shape)
print("Gray images dimensions:", get_test_set_bw(range(1, 2), [1, 2, 3], ['no_flash'])[0][0].shape)


# ------------------------------------------------- #
#               Generate imgs                       #
# ------------------------------------------------- #
def generate_img(img, cmap=plt.get_cmap('gray')):
    plt.imshow(img, cmap=cmap)
    # plt.show()


def generate_imgs(img1, img2, img3, img4):
    plt.subplot(221)
    generate_img(img1)
    plt.subplot(222)
    generate_img(img2)
    plt.subplot(223)
    generate_img(img3)
    plt.subplot(224)
    generate_img(img4)
    plt.show()


# ------------------------------------------------- #
#               Image Processing                    #
# ------------------------------------------------- #
def resize_imgs(imgs, dims=(28,28), interpolation=cv2.INTER_NEAREST):
    return np.array([cv2.resize(img, dims) for img in imgs])


# ------------------------------------------------- #
#                   Models                          #
# ------------------------------------------------- #
# load the model from disk
def load_model(filename):
    return pickle.load(open(filename, 'rb'))    

def predict_probabilities(x):
    # Flatten x
    number_pixels = x.shape[1] * x.shape[2]
    x = x.reshape((x.shape[0], number_pixels)).astype("float32")
    # Normalize
    x = x / 255
    probabilities = model.predict_proba(x)
    return probabilities


# ------------------------------------------------- #
#                   Main Program                    #
# ------------------------------------------------- #
if __name__ == "__main__":
    model = load_model(model_filename)
    # Load images
    # X_test_color, y_test = get_test_set_color(number_of_writers, number_of_digits, img_options)

    # Let us assume the black and white by now, since it is the simplest approach and it resembles the approach used for MNIST
    X_test_bw, y_test = get_test_set_bw(number_of_writers, number_of_digits, img_options)
    print(f"Loaded {len(y_test)} images")
    [print(f"\t-Loaded image with {x.shape} px") for x in X_test_bw[1:10]];

    X = X_test_bw[selected_examples]
    y = y_test[selected_examples]

    generate_imgs(*X)
    X_resized = resize_imgs(X)
    generate_imgs(*X_resized)
    
    # Expected format
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    print(f"Current format: {X_resized.shape} Vs Expected format: {X_train.shape}")
    print(f"Current format: {X_resized[0].shape} Vs Expected format: {X_train[0].shape}")

    probabilities = predict_probabilities(X_resized)
    print(probabilities)


