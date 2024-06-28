import cv2
import numpy as np
import utlis

###basic setting start
path = 'mcqpaper.jpg'
img = cv2.imread(path)

widthImage = 450
heightImage = 600
###basic setting end


img = cv2.resize(img, (widthImage,heightImage))
imgContours = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
imgCanny = cv2.Canny(imgBlur,10,50)

contours, hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contours,-1,(0,255,0),2)

imgBlank = np.zeros_like(img)

imgArray = ([img,imgGray,imgBlur,imgCanny],
            [imgContours,imgBlank,imgBlank,imgBlank])
imgStacked = utlis.stackImages(imgArray,0.5)

cv2.imshow('MCQ paper',imgStacked)
cv2.waitKey(0)















































