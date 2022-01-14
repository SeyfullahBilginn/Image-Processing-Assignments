import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
    This function is not used,
    it s for visualisation in order to comprehend process better 
"""
def draw_matches(img1, keypoints1, img2, keypoints2, matches):
    r, c = img1.shape[:2]
    r1, c1 = img2.shape[:2]

    # Create a blank image with the size of the first image + second image
    outputImg = np.zeros((max([r, r1]), c + c1, 3), dtype='uint8')
    outputImg[:r, :c, :] = np.dstack([img1, img1, img1])
    outputImg[:r1, c:c + c1, :] = np.dstack([img2, img2, img2])

    # Go over all of the matching points and extract them
    for match in matches:
        img1Idx = match.queryIdx
        img1Idx = match.trainIdx
        (x1, y1) = keypoints1[img1Idx].pt
        (x2, y2) = keypoints2[img2Idx].pt

        # Draw circles on the keypoints
        cv2.circle(outputImg, (int(x1), int(y1)), 4, (0, 255, 255), 2)
        cv2.circle(outputImg, (int(x2) + c, int(y2)), 4, (0, 255, 255), 2)

        # Connect the same keypoints
        cv2.line(outputImg, (int(x1), int(y1)), (int(x2) + c, int(y2)), (0, 0, 255), 1)

    return outputImg


# blending part
# it softens overlaying parts
def blending(dst, src, width):


    cv2.imwrite("./ddd.jpg", dst)
    cv2.imwrite("./src.jpg", src)

    h, w, _ = dst.shape

    A = src
    B = dst
    src = cv2.resize(A, (480,480))
    dst = cv2.resize(B, (480,480))


    ## generate Gaussian pyramid for src
    src_copy = src.copy()
    gp_src  = [src_copy]
    for i in range(6):
        src_copy = cv2.pyrDown(src_copy)
        gp_src.append(src_copy)

    ## generate Gaussian pyramid for dst
    dst_copy = dst.copy()
    gp_dst = [dst_copy]
    for i in range(6):
        dst_copy = cv2.pyrDown(dst_copy)
        gp_dst.append(dst_copy)

    ## generate Laplacian Pyramid for src
    src_copy = gp_src[5]
    lp_src = [src_copy]
    for i in range(5, 0, -1):
        gaussian_expanded = cv2.pyrUp(gp_src[i])
        laplacian = cv2.subtract(gp_src[i - 1], gaussian_expanded)
        lp_src.append(laplacian)

    ## generate Laplacian Pyramid for dst
    dst_copy = gp_dst[5]
    lp_dst = [dst_copy]
    for i in range(5, 0, -1):
        gaussian_expanded = cv2.pyrUp(gp_dst[i])
        laplacian = cv2.subtract(gp_dst[i - 1], gaussian_expanded)
        lp_dst.append(laplacian)

    ## Now add left and right halves of images in each level
    src_dst_pyramid = []
    n = 0
    for src_lap, dst_lap in zip(lp_src, lp_dst):
        n += 1
        cols, rows, ch = src_lap.shape
        laplacian = np.hstack((src_lap[:, 0:int(cols / 2)], dst_lap[:, int(cols / 2):]))
        src_dst_pyramid.append(laplacian)
    ## now reconstruct
    src_dst_reconstruct = src_dst_pyramid[0]
    for i in range(1, 6):
        src_dst_reconstruct = cv2.pyrUp(src_dst_reconstruct)
        src_dst_reconstruct = cv2.add(src_dst_pyramid[i], src_dst_reconstruct)

    cv2.imshow("constructed", src_dst_reconstruct)
    cv2.imwrite("./constructed.jpg", src_dst_reconstruct)

    return src_dst_reconstruct

def blending2(dst, src, width):


    h, w, _ = dst.shape
    smoothing_window = int(width / 16)
    barrier = width - int(smoothing_window / 2)

    mask1 = np.zeros((h, w))
    mask2 = np.zeros((h, w))

    offset = int(smoothing_window / 2)

    mask1[:, barrier - offset: barrier + offset + 1] = np.tile(
        np.linspace(1, 0, 2 * offset + 1).T, (h, 1)
    )
    mask1[:, : barrier - offset] = 1

    mask1 = cv2.merge([mask1, mask1, mask1])

    mask2[:, barrier - offset: barrier + offset + 1] = np.tile(
        np.linspace(0, 1, 2 * offset + 1).T, (h, 1)
    )
    mask2[:, barrier + offset:] = 1
    mask2 = cv2.merge([mask2, mask2, mask2])


    dst = dst * mask1
    src = src * mask2
    pano = src + dst


    return pano

def wrapImages(src, dst, H):
    # get height and width of images
    srcHeight, srcWidth = src.shape[:2]
    dstHeight, dstWidth = dst.shape[:2]

    # extract conners of images
    pts1 = np.float32([[0, 0], [0, srcHeight],
                       [srcWidth, srcHeight], [srcWidth, 0]]).\
        reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, dstHeight],
                       [dstWidth, dstHeight], [dstWidth, 0]]).\
        reshape(-1, 1, 2)

    pts1_ = cv2.perspectiveTransform(pts1, H)

    pts = np.concatenate((pts1_, pts2), axis=0)

    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [_, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)

    translation_dist = [-xmin, -ymin]

    widthPano = int(pts1_[3][0][0])


    heightPano = ymax - ymin

    Ht = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    wrappedSrc = cv2.warpPerspective(
        src, Ht.dot(H), (widthPano, heightPano))
    dstRZ = np.zeros((heightPano, widthPano, 3))

    dstRZ[translation_dist[1]: srcHeight + translation_dist[1], :dstWidth] = dst

    # apply blending
    pano = blending2(
            dstRZ, wrappedSrc, dstWidth
        )

    # crop black regions
    pano = crop(pano, dstHeight, pts)

    return pano

def warpTwoImages(src, dst,x):


    # detect keypoints of images
    orb = cv2.ORB_create(nfeatures=2000)

    keypoints1, descriptors1 = orb.detectAndCompute(src, None)
    keypoints2, descriptors2 = orb.detectAndCompute(dst, None)

    # find all of the matching keypoints on two images with brute force
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)

    # cv2.imshow("matches,", draw_matches())

    # Find matching points of 2 images
    matches = bf.knnMatch(descriptors1, descriptors2,k=2)
    all = []
    for m, n in matches:
        all.append(m)

    # Find the best matches
    # eliminate others
    good = []
    limit = 0.6
    for m, n in matches:
        if m.distance < limit * n.distance:
            good.append(m)

    # TO see matched points you can active this code segment
    # It is used for report
    #  convert images to grayscale
    # img1_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # img2_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # img3 = draw_matches(img1_gray, keypoints1, img2_gray, keypoints2, good[:30])
    # cv2.imshow("draw_matches.jpg" , img3)

    MIN_MATCH_COUNT = 20
    if len(good) > MIN_MATCH_COUNT:
        # Convert keypoints to an argument for findHomography
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # generate Homography matrix
        ransacReprojThreshold = 5.0
        Homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold)
        res = wrapImages(src,dst,Homography)
        return res


def stitchingMethod(list_images):

    # We assume images left to right and there will be 3 image has same objects
    # divide middle image to 2 subparts which are left and right
    left = list_images[0]
    middle = list_images[1]
    right = list_images[2]

    # LEFT PART
    leftTotal= warpTwoImages(middle,left,0)
    leftPart = leftTotal.astype("uint8")

    # RIGHT PART
    rightTotal= warpTwoImages(right,middle,1)
    rightPart = rightTotal.astype("uint8")

    # BLEND TWO PART
    total = warpTwoImages(rightPart,leftPart,2)

    return total


def crop(panorama, hDst, conners):
    [xmin, ymin] = np.int64(conners.min(axis=0).ravel() - 0.5)
    t = [-xmin, -ymin]
    conners = conners.astype(int)

    if conners[0][0][0] < 0:
        n = abs(-conners[1][0][0] + conners[0][0][0])
        panorama = panorama[t[1] : hDst + t[1], n:, :]
    else:
        if conners[2][0][0] < conners[3][0][0]:
            panorama = panorama[t[1] : hDst + t[1], 0 : conners[2][0][0], :]
        else:
            panorama = panorama[t[1] : hDst + t[1], 0 : conners[3][0][0], :]
    return panorama
