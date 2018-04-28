from PIL import Image
import numpy as np
import os

training_size = 20
img = Image.open("faces/subject01.centerlight")
width = img.size[0]
height = img.size[1]

all_img_names = os.listdir("faces/")
num_imgs = len(all_img_names)

#Vector to hold the average pixel values of the entire training set
total_img_pixels = np.zeros((width*height,1))

pixel_matrix = np.zeros((width*height,1))

i=0
for each_img in all_img_names:
    img = Image.open("faces/"+each_img)
    pixels = np.asarray(img)
    pixels = np.reshape(pixels,(width*height,1))
    total_img_pixels = pixels + total_img_pixels
    pixel_matrix = np.append(pixel_matrix,pixels,axis=1)
    i = i+1
    if i==training_size:
        break

#Delete the first column of zeros
pixel_matrix = np.delete(pixel_matrix,0,1)
total_img_pixels = total_img_pixels / num_imgs
#print(pixel_matrix)
error_matrix = pixel_matrix - total_img_pixels
#print(error_matrix)
L = np.matmul(pixel_matrix.T,pixel_matrix)
#print(L)
