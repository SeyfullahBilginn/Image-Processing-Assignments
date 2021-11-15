def colorTransfer(sourcePath,targetPath):
    import cv2
    import numpy as np
    import math
    source = cv2.imread(sourcePath)
    target = cv2.imread(targetPath)
    sourceArr = np.array(source)
    targetArr = np.array(target)

    consArr = np.array(
        [
            [0.3811, 0.5783, 0.0402],
            [0.1967, 0.7244, 0.0782],
            [0.0241, 0.1288, 0.8444]
        ]
    )

    IABArrConst = np.dot(
        [
            [1 / math.sqrt(3), 0, 0],
            [0, 1 / math.sqrt(6), 0],
            [0, 0, 1 / math.sqrt(2)]
        ]
        ,
        [
            [1, 1, 1],
            [1, 1, -2],
            [1, -1, 0]
        ]
    )

    SourceR = np.zeros(sourceArr.shape[0] * sourceArr.shape[1])
    SourceG = np.zeros(sourceArr.shape[0] * sourceArr.shape[1])
    SourceB = np.zeros(sourceArr.shape[0] * sourceArr.shape[1])

    count = 0;
    for y in range(sourceArr.shape[0]):
        for x in range(sourceArr.shape[1]):
            # print("x: ", x , " y: " , y)
            # print(sourceArr[y][x])
            # print(R[y][x])
            SourceR[count] = sourceArr[y][x][0]
            SourceG[count] = sourceArr[y][x][1]
            SourceB[count] = sourceArr[y][x][2]
            count = count + 1
    SourceRGB = np.array([SourceR, SourceG, SourceB])

    # Apply the given transformation to convert RGB source image to LMS cone space:
    LSource, MSource, SSource = np.dot(consArr, SourceRGB)

    LSource = np.where(LSource == 0, 1, LSource)
    MSource = np.where(MSource == 0, 1, MSource)
    SSource = np.where(SSource == 0, 1, SSource)

    # Convert data to logarithmic space for source:
    loggedLSource = np.log10(LSource)
    loggedMSource = np.log10(MSource)
    loggedSSource = np.log10(SSource)

    # Apply the given transformation to convert to IAB space:
    SourceI, SourceA, SourceB = np.dot(IABArrConst, np.array([loggedLSource, loggedMSource, loggedSSource]))

    # Subtract the mean of source image from the source image:
    NewSourceI = SourceI - np.mean(SourceI)
    NewSourceA = SourceA - np.mean(SourceA)
    NewSourceB = SourceB - np.mean(SourceB)

    # Variance for source IAB
    varIS = np.var(SourceI)
    varAS = np.var(SourceA)
    varBS = np.var(SourceB)

    #  ************************** TARGET

    TargetR = np.zeros(targetArr.shape[0] * targetArr.shape[1])
    TargetG = np.zeros(targetArr.shape[0] * targetArr.shape[1])
    TargetB = np.zeros(targetArr.shape[0] * targetArr.shape[1])

    count = 0;
    for y in range(targetArr.shape[0]):
        for x in range(targetArr.shape[1]):
            TargetR[count] = targetArr[y][x][0]
            TargetG[count] = targetArr[y][x][1]
            TargetB[count] = targetArr[y][x][2]
            count = count + 1
    TargetRGB = np.array([TargetR, TargetG, TargetB])

    # Apply the given transformation to convert RGB target image to LMS cone space:
    LTarget, MTarget, STarget = np.dot(consArr, TargetRGB)

    LTarget = np.where(LTarget == 0, 1, LTarget)
    MTarget = np.where(MTarget == 0, 1, MTarget)
    STarget = np.where(STarget == 0, 1, STarget)

    # Convert data to logarithmic space for target image:
    loggedLTarget = np.log10(LTarget)
    loggedMTarget = np.log10(MTarget)
    loggedSTarget = np.log10(STarget)

    # Apply the given transformation to convert to IAB space:
    TargetI, TargetA, TargetB = np.dot(IABArrConst, np.array([loggedLTarget, loggedMTarget, loggedSTarget]))

    # variance of target IAB space
    varIT = np.var(TargetI)
    varAT = np.var(TargetA)
    varBT = np.var(TargetB)

    # **************

    scaledI = (varIT / varIS) * NewSourceI
    scaledA = (varAT / varAS) * NewSourceA
    scaledB = (varBT / varBS) * NewSourceB

    # Add the target's mean to the scaled data points:
    LastI = scaledI + np.mean(TargetI)
    LastA = scaledA + np.mean(TargetA)
    LastB = scaledB + np.mean(TargetB)


    # Apply transform matrix to convert l to LMS:

    LastL, LastM, LastS = np.dot(np.dot(
        [
            [1, 1, 1],
            [1, 1, -1],
            [1, -2, 0]
        ]
        ,
        [
            [math.sqrt(3) / 3, 0, 0],
            [0, math.sqrt(6) / 6, 0],
            [0, 0, math.sqrt(2) / 2]
        ]
    ), np.array([LastI, LastA, LastB]))

    # Go back to linear space:

    L = np.power(10, LastL)
    M = np.power(10, LastM)
    S = np.power(10, LastS)

    # Apply transform matrix to convert LMS to RGB:

    R, G, B = np.dot(
        [
            [4.4679, -3.5873, 0.1193],
            [-1.2186, 2.3809, -0.1624],
            [0.0497, -0.2439, 1.2045]
        ]
        ,
        np.array([L, M, S])
    )

    # Normalised [0,255] as integer
    # it helps to get realistic colours
    # especially in the sky between clouds and house in scotland sample examples
    R = (255 * (R - np.min(R)) / np.ptp(R)).astype(int)
    G = (255 * (G - np.min(G)) / np.ptp(G)).astype(int)
    B = (255 * (B - np.min(B)) / np.ptp(B)).astype(int)

    result = np.zeros(sourceArr.shape)

    # convert RGB arrays to result image
    counter = 0;
    for y in range(result.shape[0]):
        temp = np.zeros([result.shape[1], 3])
        for x in range(result.shape[1]):
            temp[x] = [(R[counter]), (G[counter]), (B[counter])]
            counter = counter + 1

        result[y] = temp

    # save image to current working directory
    path = "./images/ColorTransferResult-" + sourcePath.split("/")[2].split(".")[0] \
           + "-" + targetPath.split("/")[2].split(".")[0] + ".jpg"
    cv2.imwrite(path, result)