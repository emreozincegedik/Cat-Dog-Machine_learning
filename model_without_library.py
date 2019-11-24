import numpy as np
import os
import cv2
import threading

class Neural_Network(object):
  def __init__(self):
    #parameters
    self.inputSize = len(X[0])
    self.outputSize = 1
    self.hiddenSize = 32
    self.hiddenSize2= 32
    self.hiddenSize3= 64

    if os.path.exists("w1.csv"):
      print("saved weights found")
      self.W1 = np.loadtxt("w1.csv",delimiter=',') 
      self.W1=self.W1.reshape(self.inputSize, self.hiddenSize)
      self.W2 = np.loadtxt("w2.csv",delimiter=',') 
      self.W2=self.W2.reshape(self.hiddenSize, self.hiddenSize2)
      self.W3 = np.loadtxt("w3.csv",delimiter=',') 
      self.W3=self.W3.reshape(self.hiddenSize2, self.hiddenSize3)
      self.W4 = np.loadtxt("w4.csv",delimiter=',') 
      self.W4=self.W4.reshape(self.hiddenSize3, self.outputSize)
    else:
      print("no saved weight")
      self.W1 = np.random.randn(self.inputSize, self.hiddenSize) 
      self.W1=self.W1 / 17500
      self.W2 = np.random.randn(self.hiddenSize, self.hiddenSize2) 
      self.W2=self.W2 / 17500
      self.W3 = np.random.randn(self.hiddenSize2, self.hiddenSize3) 
      self.W3=self.W3 / 17500
      self.W4 = np.random.randn(self.hiddenSize3, self.outputSize) 
      self.W4=self.W4 / 17500

  def forward(self, X):
    #forward propagation through our network
    self.z = X.dot(self.W1) 
    self.z2 = self.ReLU(self.z) 
    self.z3 = self.z2.dot( self.W2)
    self.z4=self.ReLU(self.z3)
    self.z5=self.z4.dot(self.W3)
    self.z6=self.ReLU(self.z5)
    self.z7=self.z6.dot(self.W4)
    o = self.pieceWise(self.z7) 
    return o
  
  def ReLU(self, x):
    return np.where(x>0, x,0.0)
  
  def dReLU(self, x):
    # return 1. * (x > 0)
    return np.where(x > 0, 1.0, 0.0)

  def pieceWise(self,x):
    return np.piecewise(x, [x < 0,x >= 0], [0, 1])
  
  def dpieceWise(self, x):
    return 1

  def sigmoid(self, s): 
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    return s * (1 - s)

  def backward(self, X, y, o):
    # backward propgate through the network
    self.o_error = y - o # error in output
    self.o_delta = self.o_error*self.dpieceWise(o) # applying derivative of pieceWise to error

    self.z6_error = self.o_delta.dot(self.W4.T) # z6 error: how much our hidden layer weights contributed to output error
    self.z6_delta = self.z6_error*self.dReLU(self.z6) # applying derivative of sigmoid to z6 error
    
    self.z4_error = self.z6_delta.dot(self.W3.T) # z4 error: how much our hidden layer weights contributed to z6 error
    self.z4_delta = self.z4_error*self.dReLU(self.z4)
    
    self.z2_error = self.z4_delta.dot(self.W2.T)
    self.z2_delta = self.z2_error*self.dReLU(self.z2)

    self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
    self.W2 += self.z2.T.dot(self.z4_delta) # adjusting second set (hidden1 --> hidden2) weights
    self.W3 += self.z4.T.dot(self.z6_delta) # adjusting third set (hidden2 --> hidden3) weights
    self.W4 += self.z6.T.dot(self.o_delta)  # adjusting fourth set (hidden3 --> output) weights


  def train (self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)

def load_images_from_folder(folder, img_size):
  images = np.array([])
  for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename),0)
    if img is not None:
      img_resize=cv2.resize(img,(img_size,img_size))
      img_resize=img_resize/255
      img=img_resize.ravel()
      # print(img)
      images=np.append(images,img)
      # print(images.size)
  images=images.reshape(int(images.size/(img_size**2)),img_size**2)
  print(f'loaded {folder} folder, found {int(images.size/(img_size**2))} images')
  return images

def shuffle_in_unison(a, b):
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def image_to_array():
  print("creating images to arrays")

  img_size=150
  from multiprocessing.pool import ThreadPool
  pool = ThreadPool(processes=2)

  dogs = pool.apply_async(load_images_from_folder, ('images/dogs', img_size)) # tuple of args for foo
  flowers = pool.apply_async(load_images_from_folder, ('images/flowers', img_size)) # tuple of args for foo

  dogs = dogs.get()  # get the return value from your function.
  flowers=flowers.get()

  all_input=np.append(dogs,flowers)

  dogs_output = np.array(([]), dtype=np.float64)
  for i in range(int(dogs.size/(img_size**2))):
    dogs_output=np.append(dogs_output,0)
  dogs_output=dogs_output.reshape(int(dogs.size/(img_size**2)),1)

  flowers_output = np.array(([]), dtype=np.float64)
  for i in range(int(flowers.size/(img_size**2))):
    flowers_output=np.append(flowers_output,1)
  flowers_output=flowers_output.reshape(int(flowers.size/(img_size**2)),1)

  all_output=np.append(dogs_output,flowers_output)
  all_input=all_input.reshape(int(all_input.size/(img_size**2)),img_size**2)
  print(all_input.size)
  print(all_output.size)

  all_input,all_output=shuffle_in_unison(all_input,all_output)

  np.savetxt('inputs.csv', all_input, delimiter=',', fmt='%f')
  np.savetxt('outputs.csv', all_output, delimiter=',', fmt='%f')

  print("created inputs, outputs.csv")


#Program starts here
if not os.path.exists("inputs.csv"):
  image_to_array()

print("Reading inputs, outputs.csv")
X=np.loadtxt("inputs.csv",delimiter=',')
y=np.loadtxt("outputs.csv",delimiter=',')
y=y.reshape(1791,1)

NN = Neural_Network()
print ("Input: \n" + str(X) )
print ("Actual Output: \n" + str(y)) 
print( "Predicted Output: \n" + str(NN.forward(X)) )
print ("Loss: \n" + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
print ("\n")

for i in range(1000000): # trains the NN 1,000,000 times
  if i%1000==0:
    print ("Loss: \n" + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
    print(f"i: {i} \n\n")
    # print (f"Loss {i}: \n" + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss

    if os.path.exists("w1.csv"):
      os.remove("w1.csv")
      os.remove("w2.csv")
      os.remove("w3.csv")
      os.remove("w4.csv")

    np.savetxt('w1.csv', NN.W1, delimiter=',', fmt='%f')
    np.savetxt('w2.csv', NN.W2, delimiter=',', fmt='%f')
    np.savetxt('w3.csv', NN.W3, delimiter=',', fmt='%f')
    np.savetxt('w4.csv', NN.W4, delimiter=',', fmt='%f')
    print("updated weights \n\n")
  NN.train(X, y)

print ("Input: \n" + str(X) )
print ("Actual Output: \n" + str(y)) 
print( "Predicted Output: \n" + str(NN.forward(X)) )
print ("Loss: \n" + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
print ("\n")
# print(str(NN.W4))
