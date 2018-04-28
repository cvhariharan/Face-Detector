from PIL import Image
import numpy as np
from numpy import linalg as LA
import os, random

training_size = 30
img = Image.open("faces/subject01.centerlight")
width = img.size[0]
height = img.size[1]

all_img_names = os.listdir("faces/")
#random.shuffle(all_img_names)
num_imgs = len(all_img_names)

#Vector to hold the average pixel values of the entire training set
total_img_pixels = np.zeros((width*height,1))

pixel_matrix = np.zeros((width*height,1))

i=0
for each_img in all_img_names:
    img = Image.open("faces/"+each_img)
    pixels = np.asarray(img)
    
    #print(pixels)
    pixels = np.reshape(pixels,(width*height,1))
    #print(pixels)
    total_img_pixels = pixels + total_img_pixels
    pixel_matrix = np.append(pixel_matrix,pixels,axis=1)
    i = i+1
    if i==training_size:
        break
#Delete the first column of zeros
pixel_matrix = np.delete(pixel_matrix,0,1)
total_img_pixels = total_img_pixels / training_size
print(total_img_pixels)
temp = np.reshape(total_img_pixels,(height,width))
im = Image.fromarray(temp)
im.show()
#print(pixel_matrix)
#print("T")
error_matrix = pixel_matrix - total_img_pixels
#print(error_matrix)
L = np.matmul(error_matrix.T,error_matrix)
L = L/training_size
print(L.shape)

#Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = LA.eig(L)
print(eigenvectors)

eigenfaces = np.matmul(error_matrix,eigenvectors)
#print(eigenfaces)
