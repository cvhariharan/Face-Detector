from PIL import Image
import numpy as np
from numpy import linalg as LA
import os, random

class FaceDectector:
    def __init__(self, training_size, sample, directory):
        self.training_size = training_size
        self.directory = directory
        img = Image.open(sample)
        self.width = img.size[0]
        self.height = img.size[1]
    
    def train(self):
        self.all_img_names = os.listdir(self.directory)
        num_imgs = len(self.all_img_names)

        #Vector to hold the average pixel values of the entire training set
        total_img_pixels = np.zeros((self.width*self.height,1))

        pixel_matrix = [] 
        i=0
        for each_img in self.all_img_names:
            img = Image.open(self.directory+each_img)
            pixels = np.asarray(img)
            pixels = np.reshape(pixels,(self.width*self.height,1))
            total_img_pixels = pixels + total_img_pixels
            pixel_matrix.append(pixels)
            i = i+1
            if i == self.training_size:
                break

        pixel_matrix = np.reshape(np.array(pixel_matrix).T,(self.width*self.height,self.training_size))
        print(pixel_matrix.shape)
        total_img_pixels = total_img_pixels / self.training_size
        self.total_img_pixels = total_img_pixels
        temp = np.reshape(total_img_pixels,(self.height,self.width))

        #Shows the average image
        #im = Image.fromarray(temp)
        #im.show()

        self.error_matrix = pixel_matrix - total_img_pixels
        L = np.matmul((self.error_matrix).T,self.error_matrix)
        L = L / self.training_size

        #Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = LA.eigh(L)

        #Take only top 20 eigenvectors with highest eigenvalues
        sort_indices = eigenvalues.argsort()[::-1]
        sort_indices = sort_indices[:20]
        eigenvectors = eigenvectors[:,sort_indices]
        self.eigenfaces = []
        for i in range(eigenvectors.shape[1]):
            self.eigenfaces.append(np.matmul(self.error_matrix,eigenvectors[:,i]))
        
        self.eigenfaces = np.array(self.eigenfaces).T
         #Contribution of the training set
        self.contribution_matrix = []
        for i in range((self.error_matrix).shape[1]):
            (self.contribution_matrix).append(np.matmul((self.eigenfaces).T,self.error_matrix[:,i]))

        self.contribution_matrix = np.array(self.contribution_matrix).T

    def detect(self,filename):
        #Classify
        test = Image.open(filename)
        test_matrix = np.asarray(test)
        test_matrix = np.reshape(test_matrix,(self.width*self.height,1))
        test_error_matrix = test_matrix - self.total_img_pixels

        omega = np.matmul((self.eigenfaces).T,test_error_matrix)

        score = self.contribution_matrix - omega
        all_scores = []
        for j in range(score.shape[1]):
            all_scores.append(LA.norm(score[:,j]))

        #print(all_scores)
        n_scores = np.array(all_scores)

        print("Detected Image: "+self.directory+" "+self.all_img_names[np.argmin(n_scores)])

        detected = Image.open(self.directory+self.all_img_names[np.argmin(n_scores)])
        detected.show()

#Example
d = FaceDectector(30,"faces/subject01.centerlight","faces/")
d.train()
d.detect("test/subject02.glasses")