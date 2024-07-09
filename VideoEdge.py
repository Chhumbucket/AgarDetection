import cv2
import time
cap = cv2.VideoCapture(0)
startTime = time.time()
interval = 10
count = 0

while True:
    ret, img = cap.read()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 10, 70)
    ret, mask = cv2.threshold(canny, 70, 255, cv2.THRESH_BINARY)
    cv2.imshow('Video feed', mask)
    
    # if time.time() - startTime >= interval:
    #     fileName = 'screenshot{}.jpg'.format(count)
    #     cv2.imwrite(fileName, mask)
    #     start_time = time.time()
    #     count += 1
    if cv2.waitKey(1) == 13:
        break
    
    
cap.release()
cv2.destroyAllWindows()