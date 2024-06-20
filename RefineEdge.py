import cv2 as cv 
import numpy as np

abs_path = '/Users/kingchhum/Desktop/Lab Materials/Edge Detection/leo.png'
img = cv.imread(abs_path) 

cv.imshow('Original', img)
cv.waitKey(0)

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
equalizedImg = cv.equalizeHist(img_gray)
blurredImage = cv.GaussianBlur(equalizedImg, (5,5), 0)

# sobelx = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
# sobely = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
# sobelxy = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
# # Display Sobel Edge Detection Images
# cv.imshow('Sobel X', sobelx)
# cv.waitKey(0)
# cv.imshow('Sobel Y', sobely)
# cv.waitKey(0)
# cv.imshow('Sobel X Y using Sobel() function', sobelxy)
# cv.waitKey(0)
# Canny Edge Detection

def auto_canny(image): 
    sigma = 0.33
    v = np.median(image) 
    
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(image, lower, upper)
    return edged


edges = cv.Canny(image=blurredImage, threshold1=30, threshold2=150)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
edges_cleaned = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
#edges_cleaned = cv.morphologyEx(edges_cleaned, cv.MORPH_ERODE, kernel) #Destroys image 

# Canny Edge Detection
# Display Canny Edge Detection Image
cv.imshow('Before Morph', edges)
cv.waitKey(0)
cv.imshow('Canny Edge Detection', edges_cleaned)
cv.waitKey(0)
cv.destroyAllWindows()

