# Python program to read image using OpenCV

# importing OpenCV(cv2) module
import cv2
import numpy as np
import math

def FloydSteingberg(image,q):

    pixel = np.array(image)

    width,height = pixel.shape

    def find_quantized_value(oldPixel):
        # takes oldPixel value
        # returns it by converting to its truncuated value
        return math.trunc(oldPixel/q)*q

    for x in range(0,width-1):
        for y in range(0,height-1):
            oldPixel = pixel[x][y]
            newPixel = find_quantized_value(oldPixel)
            pixel[x][y] = newPixel
            quant_error = (oldPixel) - (newPixel)

            pixel[x + 1][y] = pixel[x + 1][y] + quant_error * 7 / 16
            pixel[x - 1][y + 1] = pixel[x - 1][y + 1] + quant_error * 3 / 16
            pixel[x][y + 1] = pixel[x][y + 1] + quant_error * 5 / 16
            pixel[x + 1][y + 1] = pixel[x + 1][y + 1] + quant_error * 1 / 16

    # save image if u want
    path = './images/Floyd-Steinberg-' + str(q) + ".jpg"
    cv2.imwrite(path ,pixel)

    # show FloydSteingberg dithered Image
    # cv2.imshow("pixel", pixel)

    # cv2.waitKey(0)
    # Destroying present windows on screen
    # cv2.destroyAllWindows()
