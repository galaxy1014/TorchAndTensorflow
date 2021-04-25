import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras import applications

# Load mnist data
mnist = datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Data preprocessing
x_train = tf.pad(x_train, [[0,0], [2,2], [2,2]]) / 255
x_test = tf.pad(x_test, [[0, 0], [2,2], [2,2]]) / 255
x_train = tf.expand_dims(x_train, axis=3, name=None)
x_test = tf.expand_dims(x_test, axis=3, name=None)
x_train = tf.repeat(x_train, 3, axis=3)
x_test = tf.repeat(x_test, 3, axis=3)

# Load pretrained model
base_model = applications.ResNet50(include_top=False, input_shape=(32, 32, 3))

for layer in base_model.layers:
  layer.trainable = False

# Generate model
model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

# Model Evaluate
result = model.evaluate(x_test, y_test)

print('Model has {}% accuracy'.format(round(result[1],2)))