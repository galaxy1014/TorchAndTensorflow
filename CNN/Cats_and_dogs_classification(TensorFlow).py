#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


# In[2]:


train_dir = os.path.join('Cats_and_dogs/train')
val_dir = os.path.join('Cats_and_dogs/validation')


# In[3]:


train_generator = ImageDataGenerator(rescale=1./255.,
                                   rotation_range=0.4,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2,
                                   shear_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')


# In[4]:


test_generator = ImageDataGenerator(rescale=1./255.)


# In[5]:


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


# In[6]:


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


# In[7]:


model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.001, momentum=0.9),
             metrics=['accuracy'])


# In[8]:


model.summary()


# In[9]:


history = model.fit(train_data, steps_per_epoch=len(train_data),
                    validation_data=test_data, validation_steps=len(test_data), 
                    epochs=150, verbose=2)


# In[12]:


model.evaluate(test_data, verbose=5)


# In[19]:


print(train_data.class_indices)


# In[20]:


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
    
run_example()

