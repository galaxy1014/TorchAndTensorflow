import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.datasets as datasets
import numpy as np

# Load IMDB dataset
imdb = datasets.imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 25000)
x_train = pad_sequences(x_train, value=0, padding = 'pre', maxlen = 32)
x_test = pad_sequences(x_test, value=0, padding = 'pre', maxlen = 32)

# Make train, test dataset
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# Generate model
model = tf.keras.models.Sequential([
  tf.keras.layers.Embedding(25000, 100),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Dense(64, activation='tanh'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

# Training
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_ds, epochs=20, validation_data=test_ds)

# Evaluate
model.evaluate(test_ds)
