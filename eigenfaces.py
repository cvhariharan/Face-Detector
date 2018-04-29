from PIL import Image
import numpy as np
from numpy import linalg as LA
import os, random

training_size = 30
img = Image.open("faces/subject01.centerlight")
width = img.size[0]
height = img.size[1]

all_img_names = os.listdir("faces/")
#all_img_names = all_img_names[10:]
#random.shuffle(all_img_names)
num_imgs = len(all_img_names)

#Vector to hold the average pixel values of the entire training set
total_img_pixels = np.zeros((width*height,1))

pixel_matrix = [] #np.zeros((width*height,1))
i=0
for each_img in all_img_names:
    img = Image.open("faces/"+each_img)
    pixels = np.asarray(img)
    pixels = np.reshape(pixels,(width*height,1))
    total_img_pixels = pixels + total_img_pixels
    pixel_matrix.append(pixels)
    i = i+1
    if i==training_size:
        break

#Delete the first column of zeros
pixel_matrix = np.reshape(np.array(pixel_matrix).T,(width*height,training_size))#np.delete(pixel_matrix,0,1)
print(pixel_matrix.shape)
total_img_pixels = total_img_pixels / training_size
temp = np.reshape(total_img_pixels,(height,width))
#im = Image.fromarray(temp)
#im.show()
#print(pixel_matrix)
#print("T")
error_matrix = pixel_matrix - total_img_pixels
#print(error_matrix)
L = np.matmul(error_matrix.T,error_matrix)
L = L / training_size
#print(error_matrix.shape)

#Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = LA.eigh(L)

#Take only top 20 eigenvectors with highest eigenvalues
sort_indices = eigenvalues.argsort()[::-1]
sort_indices = sort_indices[:20]
eigenvectors = eigenvectors[:,sort_indices]
eigenfaces = []
for i in range(eigenvectors.shape[1]):
    eigenfaces.append(np.matmul(error_matrix,eigenvectors[:,i]))
eigenfaces = np.array(eigenfaces).T
#print(eigenfaces.shape)

#Contribution of the training set
contribution_matrix = []
for i in range(error_matrix.shape[1]):
    contribution_matrix.append(np.matmul(eigenfaces.T,error_matrix[:,i]))

#contribution_matrix = np.delete(contribution_matrix,0,1)
contribution_matrix = np.array(contribution_matrix).T
#print(contribution_matrix)
#print(all_img_names[np.argmax(LA.norm(contribution_matrix))])
#Classify
test = Image.open("test/subject01.glasses")
test_matrix = np.asarray(test)
test_matrix = np.reshape(test_matrix,(width*height,1))
test_error_matrix = test_matrix - total_img_pixels

omega = np.matmul(eigenfaces.T,test_error_matrix)

score = contribution_matrix - omega
all_scores = []
for j in range(score.shape[1]):
    all_scores.append(LA.norm(score[:,j]))

#print(all_scores)
n_scores = np.array(all_scores)

print("Detected Image: faces/"+all_img_names[np.argmin(n_scores)])

detected = Image.open("faces/"+all_img_names[np.argmin(n_scores)])
detected.show()
