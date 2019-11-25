import numpy as np
import matplotlib.pyplot as plt
import cv2
# this script is for seeing what the model sees the images as input, not necessarily needed

img_array = cv2.imread("test/download.jfif" ,cv2.IMREAD_GRAYSCALE)  # convert to array
new_array = cv2.resize(img_array, (100, 100))
plt.imshow(new_array,cmap='gray')
plt.show()
