# Python program to read image using OpenCV

# importing OpenCV(cv2) module
import cv2
import numpy as np
import pa1_2
# from matplotlib import pyplot as plt
import math
# import pa1_2 for calling Floyd-Steinberg dithering
from pa1_2 import *
img = cv2.imread('images/1.png',0)
# create numpy array for quantizedImage by initializing it to orijinal image
quantizedImage = np.array(img)

quantizeSize =  128
width,height = quantizedImage.shape

for x in range(width):
    for y in range(height):
        # assign truncuated values of every pixels to quantized Image array
        quantizedImage[x][y] = math.trunc(quantizedImage[x][y]/(quantizeSize))*(quantizeSize)

# save image if u want
path = './images/Quantized-' + str(quantizeSize) + ".jpg"
cv2.imwrite(path ,quantizedImage)

# show quantized image
# cv2.imshow('quantized', quantizedImage)

# cv2.waitKey(0)
# # Destroying present windows on screen
# cv2.destroyAllWindows()
# call FloydSteinberg dithering
pa1_2.FloydSteingberg(img,quantizeSize)
