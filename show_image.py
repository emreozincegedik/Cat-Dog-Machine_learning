import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle

img_array = cv2.imread("test/download.jfif" ,cv2.IMREAD_GRAYSCALE)  # convert to array
new_array = cv2.resize(img_array, (100, 100))
plt.imshow(new_array,cmap='gray')
plt.show()

# pickle_in = open("X.pickle","rb")
# X = pickle.load(pickle_in)
# X=X.reshape(len(X),100,100)
# pickle_in = open("y.pickle","rb")
# y = pickle.load(pickle_in)
# i=1
# # print(X[i])
# print(y[i])
# plt.imshow(X[i], cmap='gray')
# plt.show()