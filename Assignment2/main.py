import cv2
import math
import numpy as np
import timeit

imageName = "Set2"
sampleImage = cv2.imread("./SampleImages/" + imageName + ".jpg")
print(sampleImage)
sampleArr = np.array(sampleImage)
gaussianFilter = np.array(sampleImage)


def meanFilter(sampleArr,size):
    shape = sampleArr.shape
    cut = int(size/2)
    meanFilter = np.array(sampleImage)

    for y in range(cut,shape[0] - cut):
        for x in range(cut,shape[1] - cut):
            temp = sampleArr[(y-cut):(y+cut),(x-cut):(x+cut)]
            # get average value of each pixel
            avg = np.mean(temp,axis=(0,1))

            meanFilter[y][x] = avg

    return meanFilter

def gaussianKernel(x, mu, sigma):
  return math.exp( -(((x-mu)/(sigma))**2)/2.0 )

def gaussian(sampleArr,kernel_radius):

    # gaussianFilter = np.array(sampleImage)
    gaussianFilter = sampleArr
    # kernel_radius = 3 # for an 7x7 filter
    kernel_radius = int(kernel_radius/2)
    sigma = kernel_radius/2. # for [-2*sigma, 2*sigma]

    # compute the actual kernel elements
    hkernel = [gaussianKernel(x, kernel_radius, sigma) for x in range(2*kernel_radius+1)]
    vkernel = [x for x in hkernel]
    kernel2d = [[xh*xv for xh in hkernel] for xv in vkernel]

    # normalize the kernel elements
    kernelsum = sum([sum(row) for row in kernel2d])
    kernel2d = [[x/kernelsum for x in row] for row in kernel2d]
    print(kernel2d)
    for y in range(kernel_radius, len(sampleImage)-kernel_radius):
        for x in range(kernel_radius, len(sampleImage[y])-kernel_radius):
            sub = (sampleImage[y-kernel_radius:y+kernel_radius+1,x-kernel_radius:x+kernel_radius+1])
            top = [0,0,0]
            for i in range(0,kernel_radius*2+1):
                for j in range(0,kernel_radius*2+1):
                    top = top + kernel2d[i][j] * sub[i][j]
            gaussianFilter[y][x] = top
    return gaussianFilter
# https://www.cs.auckland.ac.nz/courses/compsci373s1c/PatricesLectures/Gaussian%20Filtering_1up.pdf

# 1 => 3
# gaussian(sampleArr,2)


def kuwaharaFilter(sampleImage,window_size):
    def averageRGB(pixels):

        array_1d =  pixels.transpose(2, 0, 1).reshape(3, -1)

        two = np.array(array_1d)
        size = (window_size-cut)**2

        # summation of r g b values and get average of r g b values
        r = np.sum(two, axis=1)[0] / size
        g = np.sum(two, axis=1)[1] / size
        b = np.sum(two, axis=1)[2] / size

        return [r,g,b]

    kuwaharaFilter = np.array(sampleImage)

    height,width,rgb = kuwaharaFilter.shape
    img_hsv = cv2.cvtColor(sampleImage, cv2.COLOR_BGR2HSV)

    resultingImage = sampleImage
    cut = int(window_size/2)

    maxbordery = height-cut
    maxborderx = width-cut

    for y in range(cut,maxbordery):
        for x in range(cut,maxborderx):

            lefty = x - cut

            topx = y - cut

            subs = {
                1: {"roi": img_hsv[y-cut:y+1,x:x+cut+1], "x1": y-cut, "x2": y+1, "y1": x, "y2":x+cut+1},
                2: {"roi": img_hsv[y-cut:y+1,lefty:lefty+cut+1], "x1": y-cut, "x2": y+1, "y1": lefty, "y2":lefty+cut+1},
                3: {"roi": img_hsv[topx+cut:topx+window_size,lefty:lefty+cut+1], "x1": topx+cut, "x2": topx+window_size, "y1":lefty, "y2":lefty+cut+1},
                4: {"roi": img_hsv[topx+cut:topx+window_size,lefty+cut:lefty+window_size], "x1": topx+cut, "x2": topx+window_size, "y1": lefty+cut, "y2": lefty+window_size}
            }

            std1 = np.std(subs[1]["roi"])
            std2 = np.std(subs[2]["roi"])
            std3 = np.std(subs[3]["roi"])
            std4 = np.std(subs[4]["roi"])

            vars = [std1,std2,std3,std4]

            minIndex = vars.index(min(vars)) + 1

            resultingImage[y,x] = averageRGB(sampleImage[subs[minIndex]["x1"]:subs[minIndex]["x2"],subs[minIndex]["y1"]:subs[minIndex]["y2"]])


    return resultingImage

windowSize=3

meanImage = meanFilter(sampleArr,windowSize)
cv2.imshow("Mean Image", meanImage)
cv2.imwrite("./SampleImages/meanFilter-" + imageName + "-" + str(windowSize) + ".jpg", meanImage)

# gaussianImage = gaussian(sampleArr,windowSize)
# cv2.imshow("Gaussian Image", gaussianImage)
# cv2.imwrite("./SampleImages/gaussianFilter-" + imageName + "-" + str(windowSize) + ".jpg", gaussianImage)

# kuwaharaImage = kuwaharaFilter(sampleImage, windowSize)
# cv2.imshow("Kuwahara Image", kuwaharaImage)
# cv2.imwrite("./SampleImages/kuwaharaFilter-" + imageName + "-" + str(windowSize) + ".jpg", kuwaharaImage)


# cv2.imshow("original Image", sampleImage)




cv2.waitKey(0)
cv2.destroyAllWindows
