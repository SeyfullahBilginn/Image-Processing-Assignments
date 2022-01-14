import sys
import cv2
import stitching



img1 = cv2.imread("./building1.jpg")
img2 = cv2.imread("./building2.jpg")
img3 = cv2.imread("./building3.jpg")

# images left to right
images = [img1, img2 ,img3]

panorama = stitching.stitchingMethod(images) 

cv2.imwrite("./PANORAMA.jpg", panorama)
cv2.waitKey()
