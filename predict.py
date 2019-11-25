import numpy as np
import keras
import cv2
from tqdm import tqdm
import os
from tabulate import tabulate
predict_all = True
predict_single_file_name="test/Dog/a (1).jfif" # not needed if predict_all = True

model=keras.models.load_model('models/dog_cat_2_10.h5')

test_image_directory = "test"

CATEGORIES = ["Dog", "Cat", "Random"] 
IMG_SIZE=100 # change this according to training model

def create_testing_data():
    for category in CATEGORIES:  # dogs cats

        path = os.path.join(test_image_directory,category)  # create path to dogs and cats

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                file_list.append(os.path.join(path,img)) # add file paths to file_list
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, category])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                print(e)
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

if predict_all:
  training_data = []

  file_list=[]
  create_testing_data() # convert testing images to arrays

  X = []
  y = []
  for features,label in training_data: # extract features and label
      X.append(features)
      y.append(label)

  X = np.array(X).reshape(-1,1, IMG_SIZE, IMG_SIZE, 1)
  y = np.array(y)

  X=X/255 #0-255 to 0-1
  print(file_list)

  print_table=[]

  for i in range(len(file_list)):
    classes = model.predict_classes(X[i]) # predict image (returns (1,1) array with 1 or 0)
    if classes[0][0]==1:
      prediction="cat"
    else:
      prediction="dog"
    print_table.append([prediction,y[i],file_list[i]]) # append prediction, real label and file path
  print(tabulate(print_table, headers=['prediction', 'reality','file name'])) #print all of predictions
else:
  img = cv2.imread(predict_single_file_name,cv2.IMREAD_GRAYSCALE) # read image as gray to arrays
  img = cv2.resize(img,(IMG_SIZE,IMG_SIZE)) # resize image, model can only take IMG_SIZE**2 features
  img = np.reshape(img,[1,IMG_SIZE,IMG_SIZE,1]) #model only takes this shape

  classes = model.predict_classes(img) # predict image

  if classes[0][0]==1:
    print("cat")
  else:
    print("dog")

