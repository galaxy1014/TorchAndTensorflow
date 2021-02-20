import numpy as np
import pandas as pd
import tensorflow as tf

# read csv file

df = pd.read_csv('housing.csv')
print(df.head())

print(len(df.columns))

df.columns = np.array(range(0, len(df.columns)))

print(df)

# Train data
x_data = df.iloc[:, range(len(df.columns)-1)]
print(x_data)

y_data = df.iloc[:, -1]
print(y_data)

x_train = tf.convert_to_tensor(x_data)
print(x_train)

y_train = tf.convert_to_tensor(y_data)
print(y_train)

# class data reshape
y_train = tf.reshape(y_train, [-1, 1])
print(y_train)

# Normalization
x_mean = tf.math.reduce_mean(x_train, 0)
x_std = tf.math.reduce_std(x_train, 0)
x_train -= x_mean
x_train /= x_std
print(x_train)

y_mean = tf.math.reduce_mean(y_train, 0)
y_std = tf.math.reduce_std(y_train, 0)
y_train -= y_mean
y_train /= y_std
print(y_train)

# Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, input_shape=[3]),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

print(model.summary())

# Training
model.compile(loss='mean_squared_error', 
              optimizer=tf.keras.optimizers.SGD(lr=0.001, momentum=0.9))

model.fit(x_train, y_train, epochs=1000)

# Prediction
print(model.predict([[1.46924486, -1.25935736, -0.33974768]]))