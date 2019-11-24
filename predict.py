from keras.preprocessing import image
import numpy as np
import keras
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import os
model=keras.models.load_model('models/dog_cat_2_10.h5')

IMG_SIZE=100
DATADIR = "test"

CATEGORIES = ["Dog", "Cat"]

def create_testing_data():
    for category in CATEGORIES:  # dogs cats

        path = os.path.join(DATADIR,category)  # create path to dogs and cats

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                file_list.append(os.path.join(path,img))
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, category])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                print(e)
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

training_data = []

file_list=[]
create_testing_data()

X = []
y = []
for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1,1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

X=X/255 #0-255 to 0-1
print(file_list)
from tabulate import tabulate
array=[]

classes = model.predict_classes(X[0])
array.append([classes,y[0],file_list[0]])
for i in range(len(file_list)):
  classes = model.predict_classes(X[i])
  if classes[0][0]==1:
    prediction="cat"
  else:
    prediction="dog"
  array.append([prediction,y[i],file_list[i]])
  
print(tabulate(array, headers=['prediction', 'reality','file name']))



