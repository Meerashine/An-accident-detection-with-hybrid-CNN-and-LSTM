# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 14:33:50 2020

@author: Meerashine Joe
"""


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import os, os.path
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D,LSTM,Reshape
from keras.layers import Dropout, Flatten, Dense, BatchNormalization, LeakyReLU
from keras.models import Sequential
import os, os.path
import math
import tensorflow
from tensorflow.keras.callbacks import TensorBoard


# At first we import the data that is stored and we are accessing all the 
#categories and subcategories with in the directory from both testinga and 
#training
train_categories = []
train_samples = []
for i in os.listdir(r"C:\Users\Meerashine Joe\Downloads\my projects\Dummy project\train"):
    train_categories.append(i)
    #print(type(i))
    #train_samples.append(str(len(os.listdir(r"C:\Users\Meerashine Joe\Downloads\Nitin george proj\fruits-360\Training" )))+ str(i))
    #print(train_samples)

test_categories = []
test_samples = []
for i in os.listdir(r"C:\Users\Meerashine Joe\Downloads\my projects\Dummy project\test"):
    test_categories.append(i)
    #test_samples.append(str(len(os.listdir(r"C:\Users\Meerashine Joe\Downloads\Nitin george proj\fruits-360\Test"))) + str(i))

#print(type(train_samples))
print("No. of Training Samples:", len(train_samples))
print("No. of Training Categories:", len(train_categories))
print("we are good till  now..")
#print("No. of Test Samples:", len(test_samples))
#print("No. of Testing Categories:", len(test_categories))


#import all the images in to training and testing from all the categories with indices.
train = []
test = []

print("here I goes into the crap...")
for i in os.listdir(r"C:\Users\Meerashine Joe\Downloads\my projects\Dummy project\train"):
    one_hot = np.zeros(shape=[len(train_categories)])
    #print("length",len(train_categories))
    actual_index = train_categories.index(i)
    one_hot[actual_index] = 1
    #print(one_hot)
    for files in os.listdir(r"C:\Users\Meerashine Joe\Downloads\my projects\Dummy project\train" + "\\" + i):
        #print(files)
        #for images in os.list(os.path.join(r"C:\Users\Meerashine Joe\Downloads\Nitin george proj\fruits-360\Training" +list(i), files))):
        #img_array = mpimg.imread(os.path.join(r"C:\Users\Meerashine Joe\Downloads\Nitin george proj\fruits-360\Training"  + "\\" , files))
        img_array = mpimg.imread(r"C:\Users\Meerashine Joe\Downloads\my projects\Dummy project\train" + "\\" + i +"\\" + files)
        train.append([img_array, one_hot])
        print("Train Category Status: {}/{}".format(actual_index+1, len(train_categories)))
        #print(train)
print("we are free from train")
for i in os.listdir(r"C:\Users\Meerashine Joe\Downloads\my projects\Dummy project\test"):
    one_hot = np.zeros(shape=[len(test_categories)])
    actual_index = test_categories.index(i)
    one_hot[actual_index] = 1
    for files in os.listdir(r"C:\Users\Meerashine Joe\Downloads\my projects\Dummy project\test" + "\\" +i):
        img_array = mpimg.imread(r"C:\Users\Meerashine Joe\Downloads\my projects\Dummy project\test" + "\\" + i + "\\" + files)
        test.append([img_array, one_hot])
        print("test Category Status: {}/{}".format(actual_index+1, len(test_categories)))

print("fuck I cam out successfully...")      
train_x =[]
train_y= []
test_x =[]
test_y =[]

for i in range(len(train)):
     train_x.append(train[i][0])
     train_y.append(train[i][1])

for i in range(len(test)):
    test_x.append(test[i][0])
    test_y.append(test[i][0])

#generating vaidation set from the test set and split is used to split the data.
#we take here 20 percentage.
split = np.random.choice(len(train), size=math.floor(len(train)*0.2))
print(split)


validation_x =[]
validation_y =[]

print("goes again...")
for i in range(int(len(split))):
    validation_x.append(test[i][0])
    validation_y.append(test[i][1])
 
final_train_x =np.asarray(train_x)

final_train_y =np.asarray(train_y)
#print(final_train_y.shape)
final_test_x =np.asarray(test_x)
final_test_y =np.asarray(test_y)
final_validation_x =np.asarray(validation_x)
final_validation_y =np.asarray(validation_y)


print("fasten your belts")
for i in range(len(final_train_x)):
    final_train_x[i] = (final_train_x[i] /255)
print(final_train_x.shape)
for i in range(len(final_test_x)):
    final_test_x[i] = (final_test_x[i] /255)
    
for i in range(len(final_validation_x)):
    final_validation_x[i] = (final_validation_x[i] /255)

print("check ends")

final_train_x = np.expand_dims(final_train_x, 1)
model1 =Sequential()
#model1.add(LSTM(units=3 ,input_shape=(132,3),activation="tanh", use_bias=True,recurrent_dropout = 0))

model1.add(Conv2D(input_shape=(132,3,1),kernel_size=(3,3), filters=13, padding="valid", use_bias=True))
model1.add(BatchNormalization(axis=3))
model1.add(LeakyReLU(alpha=0.01))
model1.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model1.add(Flatten())
model1.add(Dense(2,activation='softmax'))


model1.compile(loss=tf.keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])
#tensorboard = TensorBoard(log_dir="logs/{}".format("Fruits_recognition"))
print("Recognising------------------")
history = model1.fit(x = final_train_x,y = final_train_y,batch_size=32,validation_data=(final_validation_x,final_validation_y),epochs=20)    
#model.save("my model.h5")

