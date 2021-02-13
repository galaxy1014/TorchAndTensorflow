import tensorflow as tf
import tensorflow.keras.datasets as tfds

# train, test dataset 생성
mnist = tfds.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# preprocessing
x_train = x_train / 255.0
x_test = x_test / 255.0

print(x_train.shape)
print(x_test.shape)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print(x_train.shape)
print(x_test.shape)

# input layer와 output layer를 제외한 2개의 hidden layer로 구성
# input_layer : 28 * 28
# hidden_layer1 : 128
# hidden_layer2 : 64
# output_layer : 10

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=30, validation_data=(x_test, y_test))

model.evaluate(x_test, y_test, verbose=2)





