import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2
from tqdm import tqdm
import random
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

train_all_blocks_at_same_time=False #If false, you can specify which specific block to train, otherwise it will train all blocks
IMG_SIZE=100
batch_size=30 #image processing number at the same time, needed for model_fit_generator
epochs=10 #training number
training_multiplier=100 #train epochs*x time and save model x time
blocksize=10 #cross validation block size
block_number=10 #cross validation block number 
model_file='dog_cat'
model_file_name=model_file+f'_{block_number}_{blocksize}.h5'
images_folder="images"
images_file="X.pickle" # for loading images as arrays faster
images_label="y.pickle" # for loading images' labels as arrays faster
models_folder="models"
training_categories= ["Dog", "Cat"] # must be the training images folder names

def validation_block(X,y,block_size=10,validation_block_number=10):
  if validation_block_number>block_size or validation_block_number<1:
    sys.exit("block number has to be between 1-block_size!")
  if block_size<2:
    sys.exit("block size has to be greater than 1")

  # defining block's place
  train_block_divider=int(int(len(X))*(1/block_size))
  train_block=int(train_block_divider*(validation_block_number-1))
  train_block_end=int(train_block_divider*(validation_block_number))

  # block's index
  temp=np.arange(train_block,train_block_end)

  # remove block from main images array for train, block for testing
  x_train=np.delete(X,temp,0)
  x_test=X[train_block:train_block_end]

  # same for labels
  y_train=np.delete(y,temp,0)
  y_test=y[train_block:train_block_end]

  return ((x_train, y_train),(x_test,y_test))

def create_training_data():
    for category in CATEGORIES:  # dogs cats

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                print(e)
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

def create_model():
  if K.image_data_format() == 'channels_first':
      input_shape = (1, IMG_SIZE, IMG_SIZE)
  else:
      input_shape = (IMG_SIZE, IMG_SIZE, 1)

  model = Sequential()
  model.add(Conv2D(32, (3, 3), input_shape=input_shape))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(32, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(64, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Flatten())
  model.add(Dense(64))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(1))
  model.add(Activation('sigmoid'))

  model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

  return model

if os.path.exists(images_file) and os.path.exists(images_label):
  # load images as arrays if they exist as .pickle
  pickle_in = open(images_file,"rb")
  X = pickle.load(pickle_in)
  # load images' labels
  pickle_in = open(images_label,"rb")
  y = pickle.load(pickle_in)
else:
  DATADIR = images_folder

  CATEGORIES = training_categories

  training_data = []

  create_training_data() # images folder gets converted to array and appends to training_data

  random.shuffle(training_data)

  X = []
  y = []
  
  for features,label in training_data:
      X.append(features)
      y.append(label)

  X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # -1 means any number for images, training model needs this shape 
  y = np.array(y) # model also needs arrays as numpy arrays

  X=X/255 #change pixel data from 0-255 to 0-1

  # save arrays for later
  pickle_out = open(images_file,"wb")
  pickle.dump(X, pickle_out)
  pickle_out.close()

  
  pickle_out = open(images_label,"wb")
  pickle.dump(y, pickle_out)
  pickle_out.close()
  #if you want to see what the numbers are
  #np.savetxt("outputs.csv",X,delimiter=",")



if os.path.exists(os.path.join(models_folder, model_file_name)):
  model=load_model(os.path.join(models_folder, model_file_name))
else: 
  model=create_model()


# # model.fit_generator() needs this. If you need more training data use this method and comment model.fit()

# train_datagen = ImageDataGenerator(
#     # rescale=1.1,
#     # rescale=1. / 255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2)

if train_all_blocks_at_same_time:
  for i in range(blocksize):
    print(f"Training {i+1}. block of {blocksize} blocks")

    (x_train,y_train),(x_test,y_test)=validation_block(X,y,(i+1),block_number) # get current block
    model=create_model() # remake model for new block
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs) #train and test for 'epochs' times
    # # Use this instead of model.fit if you need more training data
    # model.fit_generator(
    #   train_datagen.flow(x_train, y_train, batch_size=batch_size),
    #   validation_data=(x_test, y_test), 
    #   steps_per_epoch=len(x_train) // batch_size,
    #   epochs=epochs
    #   ) 
    model.save(os.path.join(models_folder, model_file_name)) #save model for training later
else:
  (x_train,y_train),(x_test,y_test)=validation_block(X,y,blocksize,block_number) # create block's training and testing data
  for i in range(training_multiplier): #train and save for 'training_multiplier' times
    print(f"{i+1}/{training_multiplier}. training of {block_number}. block")

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs) # train for additional 'epochs' times
    # # Use this instead of model.fit() if you need more training data
    # model.fit_generator(
    #   train_datagen.flow(x_train, y_train, batch_size=batch_size),
    #   validation_data=(x_test, y_test), 
    #   steps_per_epoch=len(x_train) // batch_size,
    #   epochs=epochs
    #   ) 
    model.save(os.path.join(models_folder, model_file_name)) # save model for training later