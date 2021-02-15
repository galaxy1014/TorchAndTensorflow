#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 30k per bedroom
# house costs is 50k
# ex. 2 bedroom = 50 + (30 * 2) = 110
# create nueral network that learns this relationship 
# predict 10 bedroom house costs

import tensorflow as tf
import numpy as np


# In[4]:


# train data
x_data = np.array(range(1, 11), dtype='float32')


# In[5]:


print(x_data)


# In[6]:


y_data = np.array(np.arange(0.3, 3.0, 0.3))


# In[7]:


print(y_data)


# In[23]:


# simple model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
])


# In[24]:


model.compile(loss='mean_squared_error', optimizer='SGD', metrics=['accuracy'])


# In[25]:


model.fit(x_data, y_data, epochs=100)


# In[26]:


# 7 bedroom -> 210k
model.predict([7.0])


# In[ ]:




