#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os


# In[2]:


img = image.load_img("C:\\Users\\Seluk\\Desktop\\computer vision\\basedata\\train\\happy\\3.JPG")


# In[3]:


plt.imshow(img)


# In[4]:


cv2.imread("C:\\Users\\Seluk\\Desktop\\computer vision\\basedata\\train\\happy\\3.JPG")


# In[5]:


cv2.imread("C:\\Users\\Seluk\\Desktop\\computer vision\\basedata\\train\\happy\\3.JPG").shape


# In[6]:


train = ImageDataGenerator(rescale=1/255)
validation= ImageDataGenerator(rescale=1/255)


# In[7]:


train_dataset = train.flow_from_directory('C:\\Users\\Seluk\\Desktop\\computer vision\\basedata\\train\\',
                                         target_size=(200,200),
                                         batch_size=3,
                                         class_mode='binary')



# In[8]:


train_dataset = train.flow_from_directory('C:\\Users\\Seluk\\Desktop\\computer vision\\basedata\\train',
                                         target_size=(200,200),
                                         batch_size=3,
                                         class_mode='binary')
validation_dataset = validation.flow_from_directory('C:\\Users\\Seluk\\Desktop\\computer vision\\basedata\\train',
                                         target_size=(200,200),
                                         batch_size=3,
                                         class_mode='binary')


# In[9]:


model = tf.keras.models.Sequential([ tf.keras.layers.Conv2D(16,(3,3),activation ='relu',input_shape =(200,200,3)),
                                     tf.keras.layers.MaxPool2D(2,2),
                                     #
                                     tf.keras.layers.Conv2D(32,(3,3),activation ='relu'),
                                     tf.keras.layers.MaxPool2D(2,2),
                                     #                      
                                     tf.keras.layers.Conv2D(64,(3,3),activation ='relu'),
                                     tf.keras.layers.MaxPool2D(2,2),
                                     ##
                                     tf.keras.layers.Flatten(),
                                     ##
                                     tf.keras.layers.Dense(512,activation='relu'),
                                     ##
                                     tf.keras.layers.Dense(1,activation='sigmoid')
                                     ])


# In[10]:


model.compile(loss='binary_crossentropy',
              optimizer = RMSprop(lr=0.001),
              metrics = ['accuracy'])


# In[11]:


model_fit = model.fit(train_dataset,
                     steps_per_epoch = 3,
                     epochs = 30,
                     validation_data=validation_dataset)
                     


# In[12]:


validation_dataset.class_indices


# In[13]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os


# In[14]:


dir_path="C:\\Users\\Seluk\\Desktop\\computer vision\\basedata\\test"

for i in os.listdir(dir_path ):
    img = image.load_img(dir_path+'\\'+ i,target_size=(200,200))
    plt.imshow(img)
    plt.show() 
    
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis =0) 
    images = np.vstack([x])
    val = model.predict(images)
    if val == 0:
        print("THE PERSON IN IMAGE IS SAD")
    else: 
        print("THE PERSON IN IMAGE IS happy")
    
 


# In[19]:


model.save('model_save')


# In[20]:


model.save('my_model.h5')


# In[23]:


# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model as a SavedModel.
get_ipython().system('mkdir -p saved_model')
model.save('saved_model/my_model')


# In[22]:


tf.keras.callbacks.ModelCheckpoint('C:\\Users\\Seluk\\Desktop\\computer vision\\basedata\\test',
   filepath ,
    monitor='val_loss',
    verbose=0,
    save_best_only=False,
    save_weights_only=False,
    mode='auto',
    save_freq='epoch',
    options=None,
    initial_value_threshold=None,
    **kwargs
)


# In[19]:


checkpoint_path = "C:\\Users\\Seluk\\Desktop\\computer vision\\basedata\\train\\cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
model = create_model()

# Train the model with the new callback
model.fit(train_images, 
          train_labels,  
          epochs=30,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  # Pass callback to training

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.


# In[20]:


loaded = tf.saved_model.load('C:\\Users\\Seluk\\Desktop\\computer vision\\basedata\\tensorflow 4cd.h5')
print(list(loaded.signatures.keys()))  # ["serving_default"]


# In[ ]:





# In[20]:


model = tf.keras.models.Sequential ([
     keras.layers.Dense(512,activation = "relu"),
     input_shape = (784,)),
     keras.layers.dropout(0.2),
     keras.layers.dense(10)
])


# In[ ]:





# In[21]:


train_dataset.class_indices


# In[36]:


train_dataset.classes


# In[37]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Sequential
from tensorflow import keras
import os

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


model = tf.keras.utils.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='spare_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)
model.save('handwritten.model')


# In[1]:


import os

import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)


# In[2]:


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# In[3]:


# Define a simple sequential model
def create_model():
  model = tf.keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  return model

# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()


# In[4]:


checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model with the new callback
model.fit(train_images, 
          train_labels,  
          epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  # Pass callback to training

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.


# In[5]:


os.listdir(checkpoint_dir)


# In[6]:


# Create a basic model instance
model = create_model()

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))


# In[7]:


# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 32

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=5*batch_size)

# Create a new model instance
model = create_model()

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

# Train the model with the new callback
model.fit(train_images, 
          train_labels,
          epochs=50, 
          batch_size=batch_size, 
          callbacks=[cp_callback],
          validation_data=(test_images, test_labels),
          verbose=0)


# In[8]:


os.listdir(checkpoint_dir)


# In[9]:


latest = tf.train.latest_checkpoint(checkpoint_dir)
latest


# In[10]:


# Create a new model instance
model = create_model()

# Load the previously saved weights
model.load_weights(latest)

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


# In[11]:


# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model as a SavedModel.
get_ipython().system('mkdir -p saved_model')
model.save('saved_model/my_model')


# In[18]:


# my_model directory
get_ipython().system('ls saved_model')

# Contains an assets folder, saved_model.pb, and variables folder.
get_ipython().system('ls saved_model/my_model')


# In[13]:


new_model = tf.keras.models.load_model('saved_model/my_model')

# Check its architecture
new_model.summary()


# In[14]:


# Evaluate the restored model
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

print(new_model.predict(test_images).shape)


# In[15]:


# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model should be saved to HDF5.
model.save('my_model.h5')


# In[16]:


# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('my_model.h5')

# Show the model architecture
new_model.summary()


# In[17]:


loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))


# In[ ]:




