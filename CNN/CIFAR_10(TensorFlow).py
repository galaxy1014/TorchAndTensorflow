#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import datasets


# In[2]:


cifar_10 = datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar_10.load_data()


# In[3]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[4]:


# preprocessing
x_train = x_train / 255.0
x_test = x_test / 255.0


# In[5]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


# In[6]:


model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9),
             metrics=['accuracy'])


# In[7]:


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if(logs.get('accuracy') >= 0.98):
            print('\nReached accuracy 98%\n')
            self.model.stop_training = True
            
callback = MyCallback()


# In[8]:


model.fit(x_train, y_train, epochs=1000, batch_size=32, shuffle=True, callbacks=[callback])


# In[57]:


import numpy as np

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

predict_number = int(np.random.randint(0, 10000, 1))

def prediction():
    predict_class = model.predict_classes(x_test)[predict_number]
    real_class = int(y_test[predict_number])
    
    for i in range(0, len(labels)):
        if predict_class == i:
            print('predict class is : ', labels[i])
        if real_class == i:
            print('real class is : ', labels[i])


prediction()


# In[ ]:




