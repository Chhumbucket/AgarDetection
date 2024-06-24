import cv2 as cv
import numpy as np

# Load the image
image_path = '/Users/kingchhum/Desktop/Lab Materials/Edge Detection/leo.png'
img = cv.imread(image_path)

# Check if the image is loaded successfully
if img is None:
    print("Error: Could not load image.")
    exit()

# Convert to grayscale, equalize histogram, and apply Gaussian Blur
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blurredImage = cv.GaussianBlur(cv.equalizeHist(img_gray), (5, 5), 0)

# Perform Canny edge detection and clean edges using morphological operations
edges = cv.Canny(blurredImage, threshold1=30, threshold2=150)
edges_cleaned = cv.morphologyEx(edges, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (2, 2)))

# Find contours
contours, _ = cv.findContours(edges_cleaned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# List to store the points
points = []

# Variable to toggle edge detection view
show_edges = False

# Mouse callback function to store the points and calculate distance
def click_event(event, x, y, flags, param):
    global points
    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x, y))

def lengthConverter(pixels):
    pixels_per_mm = 3.78
    return pixels / pixels_per_mm

def areaConverter(pixels_area):
    pixels_per_mm = 3.78
    return pixels_area / (pixels_per_mm ** 2)

def calculate_manual_area(pts):
    if len(pts) < 3:
        return 0
    pts = np.array(pts, np.int32)
    return cv.contourArea(pts)

# Set the mouse callback function to the window
cv.imshow('Processed Image', img)
cv.setMouseCallback("Processed Image", click_event)

while True:
    if show_edges:
        temp_img = cv.cvtColor(edges_cleaned, cv.COLOR_GRAY2BGR)
    else:
        temp_img = img.copy()
    
    # Draw the manually selected points
    for point in points:
        cv.circle(temp_img, point, 5, (255, 0, 0), -1)

    # Draw the manually selected area
    if len(points) > 1:
        cv.polylines(temp_img, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)
    
    if len(points) >= 3:
        manual_area_pixels = calculate_manual_area(points)
        manual_area_mm = areaConverter(manual_area_pixels)
        cv.putText(temp_img, f'Manual Area: {manual_area_mm:.2f} mm^2', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Calculate and display the area of the largest contour
    if contours:
        largest_contour = max(contours, key=cv.contourArea)
        contour_area_pixels = cv.contourArea(largest_contour)
        contour_area_mm = areaConverter(contour_area_pixels)
        cv.putText(temp_img, f'Contour Area: {contour_area_mm:.2f} mm^2', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv.drawContours(temp_img, [largest_contour], -1, (0, 255, 0), 2)
        
        # Draw the outer edge contour
        outer_edge_contour = np.vstack([contour for contour in contours]).squeeze()
        outer_edge_contour = cv.convexHull(outer_edge_contour)
        outer_edge_area_pixels = cv.contourArea(outer_edge_contour)
        outer_edge_area_mm = areaConverter(outer_edge_area_pixels)
        cv.drawContours(temp_img, [outer_edge_contour], -1, (255, 0, 0), 2)
        cv.putText(temp_img, f'Outer Edge Area: {outer_edge_area_mm:.2f} mm^2', (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv.imshow("Processed Image", temp_img)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('e'):
        show_edges = not show_edges

cv.destroyAllWindows()
