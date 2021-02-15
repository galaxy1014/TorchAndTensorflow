import tensorflow as tf
import numpy as np

# train data
x_data = np.random.randint(1, 100, size=15)
x_data = x_data.reshape(-1, 3)
x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)
print(x_data)

y_data = np.array([[152], [185], [180], [196], [142]], dtype='float32')
y_data = y_data.reshape(-1, 1)
print(y_data)

# Simple Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=[3])
])


model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.SGD(lr=1e-5), metrics=['accuracy'])
model.fit(x_data, y_data, epochs=2000)


print(model.predict(np.array([[71, 23, 82]], dtype='float32')))

