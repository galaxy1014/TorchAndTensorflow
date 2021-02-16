# Classfying Fashion MNIST using single CNN
import tensorflow as tf
import tensorflow.keras.datasets as datasets

# data generation
f_mnist = datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = f_mnist.load_data()

print(x_train.shape)
print(x_test.shape)

# preprocessing
x_train = x_train / 255.0
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test / 255.0
x_test = x_test.reshape(10000, 28, 28, 1)

print(x_train.shape)
print(x_test.shape)

# model 
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# loss : sparse_categorical_crossentropy
# optimizer : Adam
model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Training
model.fit(x_train, y_train, epochs=30, validation_data=(x_test, y_test), verbose=5)

# Model evaluation
model.evaluate(x_test, y_test, verbose=2)

import numpy as np
from matplotlib import pyplot as plt

n = np.random.randint(0, 10, size=1)
plt.imshow(x_test[n].reshape(28, 28), cmap='Greys', interpolation='nearest')
plt.show()

print('The Answer is ', model.predict_classes(x_test[n].reshape((1, 28, 28, 1))))


