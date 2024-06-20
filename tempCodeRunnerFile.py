import cv2 as cv
import numpy as np

abs_path = '/Users/kingchhum/Desktop/Lab Materials/Edge Detection/leo.png'
img = cv.imread(abs_path, cv.IMREAD_UNCHANGED)

if img.shape[2] == 4:
    bgr = img[:, :, :3]
    alpha = img[:, :, 3]
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 55, 255])
    mask = cv.inRange(hsv, lower_white, upper_white)
    mask_inv = cv.bitwise_not(mask)
    bgr[mask_inv != 0] = [0, 0, 0]
    result = cv.merge((bgr, alpha))
else:
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 55, 255])
    mask = cv.inRange(hsv, lower_white, upper_white)
    mask_inv = cv.bitwise_not(mask)
    img[mask_inv != 0] = [0, 0, 0]
    result = img

cv.imshow('Non-white areas filled with black', result)
cv.waitKey(0)
cv.destroyAllWindows()
