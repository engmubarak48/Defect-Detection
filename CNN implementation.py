# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 04:09:58 2019

@author: Jama Hussein Mohamud
"""
'''
data/
    train/
        neg/
            0579.PNG
            0581.PNG
            ...
        pos/
            0576.PNG
            0576.PNG
            ...
    test data/
            test/
              0021.PNG
              0027.PNG
              0001.PNG
              0002.PNG
            ...
'''

from keras.applications.resnet50 import preprocess_input
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, applications
import matplotlib.pyplot as plt

#%%


train_data_dir = "data"
test_data_dir = 'test data'
img_height = 224
img_width = 224
batch_size = 16


train_datagen =  ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=90,
      horizontal_flip=True,
      vertical_flip=True,
      validation_split=0.2
    )

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation')

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'  # only data, no labels
        )  # keep data in same order as labels


model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
for layer in model.layers[:15]:
    layer.trainable = False

#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(1, activation="sigmoid")(x)

# creating the final model 
finetune1_model = Model(inputs = model.input, outputs = predictions)

# compile the model 
finetune1_model.compile(loss = "binary_crossentropy", optimizer = optimizers.SGD(lr=0.0001), metrics=["accuracy"])

#%%
# Train and save the model weights

NUM_EPOCHS = 30
num_train_images = 575


filepath= "VGG19" + "_model_weights.h5"

history = finetune1_model.fit_generator(train_generator, epochs=NUM_EPOCHS, validation_steps=8, 
                                       steps_per_epoch=num_train_images // batch_size, 
                                       validation_data=validation_generator)

finetune1_model.save_weights(filepath)

#%%
# visualizing losses and accuracy
train_loss=history.history['loss']
val_loss=history.history['val_loss']
train_acc=history.history['acc']
val_acc=history.history['val_acc']
xc=range(NUM_EPOCHS)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

#%%


















