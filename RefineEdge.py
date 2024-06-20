import cv2 as cv
import numpy as np

abs_path = '/Users/kingchhum/Desktop/Lab Materials/Edge Detection/leo.png'
img = cv.imread(abs_path)

# Convert to grayscale, equalize histogram, and apply Gaussian Blur
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blurredImage = cv.GaussianBlur(cv.equalizeHist(img_gray), (5, 5), 0)

# Perform Canny edge detection and clean edges using morphological operations
edges = cv.Canny(blurredImage, threshold1=30, threshold2=150)
edges_cleaned = cv.morphologyEx(edges, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (2, 2)))

# Find contours and draw them on the original image
contours, _ = cv.findContours(edges_cleaned, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#cv.drawContours(img, contours, -1, (0, 255, 0), 3)

# List to store the points
points = []

# Mouse callback function to store the points and calculate distance
def click_event(event, x, y, flags, param):
    global points
    if event == cv.EVENT_LBUTTONDOWN:
        if len(points) == 2:
            points.clear()  # Clear points when two points are already clicked
        points.append((x, y))

# Set the mouse callback function to the window
cv.imshow('Processed Image', img)
cv.setMouseCallback("Processed Image", click_event)

while True:
    temp_img = img.copy()
    
    if len(points) == 2:
        # Draw a line between the points
        cv.line(temp_img, points[0], points[1], (0, 0, 255), 2)
        # Calculate the distance
        distance = np.sqrt((points[0][0] - points[1][0])**2 + (points[0][1] - points[1][1])**2)
        # Display the distance
        midpoint = ((points[0][0] + points[1][0]) // 2, (points[0][1] + points[1][1]) // 2)
        cv.putText(temp_img, f'{distance:.2f} pixels', midpoint, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    for point in points:
        cv.circle(temp_img, point, 5, (255, 0, 0), -1)
    
    cv.imshow("Processed Image", temp_img)

    # Break the loop when 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
