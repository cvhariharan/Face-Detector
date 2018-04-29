import eigenfaces

"""
Parameters 
----------- 
1.training_size - specifies the number of images to be used for training
2.sample - sample image from the training set to know parameters such as width and height of the images in the training set
3.directory - directory containing the training images
-----------
"""
detector = eigenfaces.FaceDectector(30,"faces/subject01.centerlight","faces/")

#Trains the detector
detector.train()

#Searches for the image closest to the input image in the training set. Shows the image if found.
detector.detect("test/subject02.centerlight")