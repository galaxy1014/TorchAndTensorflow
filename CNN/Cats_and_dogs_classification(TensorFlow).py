import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


# train, test Data
train_dir = os.path.join('Cats_and_dogs/train')
val_dir = os.path.join('Cats_and_dogs/validation')

# Data Augmentation

train_generator = ImageDataGenerator(rescale=1./255.,
                                   rotation_range=0.4,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2,
                                   shear_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_generator = ImageDataGenerator(rescale=1./255.)

train_data = train_generator.flow_from_directory(train_dir,
                                                batch_size=64,
                                                shuffle=True,
                                                target_size=(150, 150),
                                                class_mode='binary')

test_data = test_generator.flow_from_directory(val_dir,
                                                batch_size=64,
                                                shuffle=True,
                                                target_size=(150, 150),
                                                class_mode='binary')

# Model

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.001, momentum=0.9),
             metrics=['accuracy'])

print(model.summary())

# training

history = model.fit(train_data, steps_per_epoch=len(train_data),
                    validation_data=test_data, validation_steps=len(test_data), 
                    epochs=150, verbose=2)

# Evaluating
model.evaluate(test_data, verbose=5)

print(train_data.class_indices)

# Testing

from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_image(filename):
    img = load_img(filename, target_size=(150, 150))
    img = img_to_array(img)
    img = img.reshape(1, 150, 150, 3)
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img

def run_example():
    img = load_image('sample_image.jpg')
    result = model.predict(img)
    if result[0] == 0:
        print('cat')
    else:
        print('dog')
    
print(run_example())

