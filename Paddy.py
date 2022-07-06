#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from keras import Sequential
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


# In[3]:


img = image.load_img(r"C:\Users\sanga\OneDrive\Desktop\Project\Training\Healthy\100025.jpg")


# In[4]:


plt.imshow(img)


# In[5]:


cv2.imread(r"C:\Users\sanga\OneDrive\Desktop\Project\Training\Healthy\100025.jpg").shape


# In[6]:


train = ImageDataGenerator(rescale=1/190)


# In[7]:


validation = ImageDataGenerator(rescale=1/190)


# In[8]:


train_dataset = train.flow_from_directory(r"C:\Users\sanga\OneDrive\Desktop\Project\Training" ,
                                          target_size=(200,200),batch_size=3,class_mode="binary")


# In[9]:


train_dataset = train.flow_from_directory(r'C:\Users\sanga\OneDrive\Desktop\Project\Training',
                                          target_size=(200,200),batch_size=3,class_mode="binary")
validation_dataset = validation.flow_from_directory(r'C:\Users\sanga\OneDrive\Desktop\Project\Validation',
                                          target_size=(200,200),batch_size=3,class_mode="binary")


# In[10]:


train_dataset.class_indices
validation_dataset.class_indices


# In[11]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(200,200,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
      
])


# In[12]:


model.compile(loss='binary_crossentropy',
             optimizer= RMSprop(lr=0.001),
             metrics=['accuracy'])


# In[13]:


model_fit= model.fit(train_dataset,
                    steps_per_epoch=3,
                     epochs=10,
                     validation_data = validation_dataset)


# In[14]:


dir_path = r'D:\Testing\Unhealthy'
for i in os.listdir(dir_path):
    img = image.load_img(dir_path+'\\'+i,target_size=(200,200))
    plt.imshow(img)
    plt.show()
    
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    images=np.vstack([x])
    val = model.predict(images)
    if val == 0:
        print("Healthy Paddy")
    else:
        print("Unhealthy Paddy")


# In[ ]:




