import cv2 as cv 
import numpy as np

abs_path = '/Users/kingchhum/Desktop/Lab Materials/Edge Detection/leo.png'
img = cv.imread(abs_path) 

cv.imshow('Original', img)
cv.waitKey(0)

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
equalizedImg = cv.equalizeHist(img_gray)
blurredImage = cv.GaussianBlur(equalizedImg, (5,5), 0)



edges = cv.Canny(image=blurredImage, threshold1=30, threshold2=150)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
edges_cleaned = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
#edges_cleaned = cv.morphologyEx(edges_cleaned, cv.MORPH_ERODE, kernel) #Destroys image 

# Canny Edge Detection
# Display Canny Edge Detection Image
contours, hierarchy = cv.findContours(edges_cleaned, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cnt = contours[4]
cv.drawContours(img, contours , -1, (0,255,0), 3)
cv.imshow('Edge', img)
cv.waitKey(0)
cv.imshow('Before Morph', edges)
cv.waitKey(0)
cv.imshow('Canny Edge Detection', edges_cleaned)
cv.waitKey(0)
cv.destroyAllWindows()

#Since the image is transparancent, maybe lets do it where if it is slightly associated with the color
#We color it in 