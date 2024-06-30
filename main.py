import cv2
import numpy as np
import utlis

###basic setting start
path = 'mcqpaper.jpg'
img = cv2.imread(path)

widthImage = 600  # Increase size for better resolution
heightImage = 800  # Increase size for better resolution
###basic setting end

img = cv2.resize(img, (widthImage, heightImage))
imgContours = img.copy()
imgBiggestContours = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
imgCanny = cv2.Canny(imgBlur, 10, 50)

contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 2)

# find rect angle in image
rectCon = utlis.rectContour(contours)

if len(rectCon) >= 2:
    biggestContour = utlis.getCornerPoints(rectCon[0])
    gradePoints = utlis.getCornerPoints(rectCon[1])

    if biggestContour.size != 0:
        biggestContour = utlis.reorder(biggestContour)
        cv2.drawContours(imgBiggestContours, [biggestContour], -1, (0, 255, 0), 20)
    if gradePoints.size != 0:
        gradePoints = utlis.reorder(gradePoints)
        cv2.drawContours(imgBiggestContours, [gradePoints], -1, (255, 0, 0), 20)

imgBlank = np.zeros_like(img)

imgArray = ([img,imgCanny,imgContours, imgBiggestContours])
imgStacked = utlis.stackImages(imgArray, 0.5)

cv2.imshow('MCQ paper', imgStacked)
cv2.waitKey(0)
