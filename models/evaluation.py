import numpy as np
import cv2
from tqdm import tqdm
import sys
import pickle
from keras.models import Sequential, load_model
from keras import backend as K
from tabulate import tabulate
model_name="dog_cat"
block_size=10

def validation_block(X,y,block_size=10,validation_block_number=10):
  if validation_block_number>block_size or validation_block_number<1:
    sys.exit("block number has to be between 1-block_size!")
  if block_size<2:
    sys.exit("block size has to be greater than 1")

  train_block_divider=int(int(len(X))*(1/block_size))
  train_block=int(train_block_divider*(validation_block_number-1))
  train_block_end=int(train_block_divider*(validation_block_number))

  temp=np.arange(train_block,train_block_end)

  x_train=np.delete(X,temp,0)
  x_test=X[train_block:train_block_end]

  y_train=np.delete(y,temp,0)
  y_test=y[train_block:train_block_end]

  return ((x_train, y_train),(x_test,y_test))

try:
  pickle_in = open("X.pickle","rb")
  X = pickle.load(pickle_in)

  pickle_in = open("y.pickle","rb")
  y = pickle.load(pickle_in)
except:
  pickle_in = open("../X.pickle","rb")
  X = pickle.load(pickle_in)

  pickle_in = open("../y.pickle","rb")
  y = pickle.load(pickle_in)



evaluation=[]
for i in range(block_size):
  (x_train, y_train), (x_test,y_test)=validation_block(X,y,block_size,(i+1))
  model=load_model(f"{model_name}_{i+1}_{block_size}.h5")
  evaluation.append(model.evaluate(x_test,y_test))
  evaluation[i].append(f"{model_name}_{i+1}_{block_size}.h5")
print(tabulate(evaluation, headers=['loss','accuracy', 'model name']))