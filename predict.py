from keras.preprocessing import image
import numpy as np
import keras
import matplotlib.pyplot as plt
import cv2
model=keras.models.load_model('dog_cat_2_10.h5')

# img_array = cv2.imread("test/a (5).jfif" ,cv2.IMREAD_GRAYSCALE)  # convert to array
# new_array = cv2.resize(img_array, (100, 100))
# img=np.array(new_array)
# img=np.expand_dims(img, axis=0)
# img=np.expand_dims(img, axis=-1)
# # img=np.moveaxis(img,0,-1)

# print(img.shape)
# print(img)

img = cv2.imread("test/download.jfif",cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img,(100,100))
img = np.reshape(img,[1,100,100,1])

classes = model.predict_classes(img)




# img_pred=image.load_img('test/a (9).jfif', target_size=(100,100),color_mode='grayscale')
# img_pred=np.array(img_pred)
# img_pred=np.expand_dims(img_pred, axis=0)
# print(img_pred.shape)
# images = np.vstack([img_pred])
# print(images.shape)
# classes = model.predict_classes(images, batch_size=10)
# print(classes)
if classes[0][0]==1:
  print("cat")
else:
  print("dog")