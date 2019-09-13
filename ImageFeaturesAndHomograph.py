import numpy as np
import cv2 as cv

UBIT = "sgotherw"
np.random.seed(sum([ord(c) for c in UBIT]))

FOLDER_PATH = ''

def ReadImage(image,color = 1):
    #read colored image
    img_color = cv.imread(FOLDER_PATH + image,color)
    return img_color

def WriteImage(imageName, image):    
    cv.imwrite(FOLDER_PATH + imageName,image)
    return

def ConvertToGrayScale(image):
    img_gray= cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    return img_gray

#Detect SIFT keypoints
def FindSIFTKeypointsAndDescriptors(image,img_gray, outputImageName):      
    sift = cv.xfeatures2d.SIFT_create()
    #finds the keypoint in the image
    keyPoints,descriptors = sift.detectAndCompute(img_gray,None)
    img_output = cv.drawKeypoints(image,keyPoints,None)
    WriteImage(outputImageName,img_output)
    return keyPoints,descriptors

def MatchKeyPoints(descriptors1,descriptors2, k = 2):    
    bf = cv.BFMatcher()
    matches = bf.knnMatch(descriptors1,descriptors2, k=k)      
    return matches

def GetGoodMatches(descriptors1,descriptors2, k = 2):    
    matches = MatchKeyPoints(descriptors1,descriptors2, k)
    # filter good matches
    goodMatches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            goodMatches.append(m)
    return goodMatches

def FindHomographyMatrix(goodMatches):
    sourcePoints = np.float32([ keyPoints1[m.queryIdx].pt for m in goodMatches ]).reshape(-1,1,2)
    destinationPoints = np.float32([ keyPoints2[m.trainIdx].pt for m in goodMatches ]).reshape(-1,1,2)
    homographyMatrix, mask = cv.findHomography(sourcePoints, destinationPoints, cv.RANSAC,5.0)
    print('Homography Matrix:')
    print(homographyMatrix)
    return homographyMatrix, mask

def GetRandomInliers(goodMatches,matchesMask,mask,numberOfSamples): 
    inliersIndexes = np.where(np.asarray(mask.ravel()) == 1)[0]
    randomInliersIndexes = np.random.choice(inliersIndexes,numberOfSamples)
    randomMatchesMask = [matchesMask[i] for i in randomInliersIndexes]
    randomGoodMatches = [goodMatches[i] for i in randomInliersIndexes]    
    return randomMatchesMask, randomGoodMatches

def GetImageDimensions(image):
    return image.shape[:2]

def PerspectiveTransform(height1,width1,height2,width2,homographyMatrix):
    points1 = np.float32([[0,0],
                          [0,height1],
                          [width1,height1],
                          [width1,0]]).reshape(-1,1,2)
    points2 = np.float32([[0,0],
                          [0,height2],
                          [width2,height2],
                          [width2,0]]).reshape(-1,1,2)
    points2_transformed = cv.perspectiveTransform(points2, homographyMatrix)
    points_concatenate = np.concatenate((points1, points2_transformed), axis=0)
    return points_concatenate

def WarpAndStitchImage(points,height1,width1,image1,image2,homographyMatrix):
    
    [height_min, width_min] = np.int32(points.min(axis=0).ravel())
    [height_max, width_max] = np.int32(points.max(axis=0).ravel())
    
    dimension = [-height_min,-width_min]
    
    H_translate = np.array([[1,0,dimension[0]],
                            [0,1,dimension[1]],
                            [0,0,1]])    
    result = cv.warpPerspective(image2, 
                                H_translate.dot(homographyMatrix), 
                                (height_max-height_min, width_max-width_min))
    result[dimension[1]:height1+dimension[1],
           dimension[0]:width1 +dimension[0]] = image1
    return result

def CreatePanorama(image1, image2, homographyMatrix):
    height1,width1 = GetImageDimensions(image1)
    height2,width2 = GetImageDimensions(image2)
    points = PerspectiveTransform(height1,width1,height2,width2,homographyMatrix)
    result = WarpAndStitchImage (points,height1,width1,image1,image2,homographyMatrix)    
    return result

img1_color = ReadImage('mountain1.jpg')
img2_color = ReadImage('mountain2.jpg')

img1_gray = ReadImage('mountain1.jpg',0) 
img2_gray = ReadImage('mountain2.jpg',0)

# find the keypoints and descriptors with SIFT
keyPoints1, descriptors1 = FindSIFTKeypointsAndDescriptors(img1_color,img1_gray,'task1_sift1.jpg')
keyPoints2, descriptors2 = FindSIFTKeypointsAndDescriptors(img2_color,img2_gray,'task1_sift2.jpg')

goodMatches = []
goodMatches = GetGoodMatches(descriptors1,descriptors2, k=2)

img_drawMatches = cv.drawMatches(img1_color,keyPoints1,img2_color,keyPoints2,goodMatches,None,flags = 0)

WriteImage('task1_matches_knn.jpg',img_drawMatches)

homographyMatrix, mask = FindHomographyMatrix(goodMatches)

matchesMask = mask.ravel().tolist()

randomMatchesMask, randomGoodMatches = GetRandomInliers(goodMatches,matchesMask,mask,numberOfSamples = 10)

draw_parameters = dict(matchColor = (255,0,255),singlePointColor = None,matchesMask = randomMatchesMask,flags = 2)

img_inliers = cv.drawMatches(img1_color,keyPoints1,img2_color,keyPoints2,randomGoodMatches,None,**draw_parameters)

WriteImage('task1_4.jpg',img_inliers)

img_panorama = CreatePanorama(img2_color,img1_color,homographyMatrix)

cv.imwrite(FOLDER_PATH + 'task1_pano.jpg',img_panorama)