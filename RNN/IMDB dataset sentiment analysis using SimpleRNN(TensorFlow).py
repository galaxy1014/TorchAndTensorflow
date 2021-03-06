import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.datasets as datasets

# Load IMDB dataset
imdb = datasets.imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 10000)
x_train = pad_sequences(x_train, value=0, padding = 'pre', maxlen = 32)
x_test = pad_sequences(x_test, value=0, padding = 'pre', maxlen = 32)

# Make train, test dataset
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# Generate model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(10000, 16),
    tf.keras.layers.SimpleRNN(32),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# Training
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(train_ds, epochs=20, validation_data=test_ds)