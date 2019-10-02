from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()

number_samples = X_train.shape[0]
number_pixels = X_train.shape[1] * X_train.shape[2]
print(f"X_train.shape={X_train.shape}")
print(f"Number of images: {number_samples}\nNumber of pixels: {number_pixels}")

# Flatten 28*28 images to a 784 vector for each image
X_train = X_train.reshape((X_train.shape[0], number_pixels)).astype("float32")
X_test = X_test.reshape((X_test.shape[0], number_pixels)).astype("float32")

# Normalize inputs form 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# One-hot encode the class values
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

number_classes = y_test.shape[1]

def baseline_model():
    # Create model
    model = Sequential()
    model.add(Dense(number_pixels, input_dim=number_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(number_classes, kernel_initializer='normal', activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = baseline_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)

print("Baseline error: %.2f%%" % (100 - scores[1]*100))

