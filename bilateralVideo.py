import cv2
from skimage import filters, measure, morphology, draw
import numpy as np

def highlight_significantly_longer_tips_video(video_path, output_path, line_position_microns, pixel_to_micron_conversion, std_factor=2):
    # Convert line position from microns to pixels
    line_position_pixels = line_position_microns * pixel_to_micron_conversion
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform edge detection on the frame
        detectedEdges = filters.sobel(gray)

        # Threshold the edge image to create a binary image
        binaryImage = detectedEdges > filters.threshold_otsu(detectedEdges)

        # Label the connected regions
        labeledImage, numRegions = measure.label(binaryImage, return_num=True, background=0)

        # Calculate the properties of the labeled regions
        properties = measure.regionprops(labeledImage)

        # Remove small regions which are considered as noise
        minRegionSize = 100  # Adjust this value based on the noise size
        cleanImage = morphology.remove_small_objects(labeledImage, min_size=minRegionSize)

        # Calculate the average and standard deviation of the major axis lengths
        major_axis_lengths = [prop.major_axis_length for prop in properties if prop.area >= minRegionSize]
        mean_length = np.mean(major_axis_lengths)
        std_length = np.std(major_axis_lengths)
        threshold_length = mean_length + std_factor * std_length

        # Identify significantly longer tips
        significantly_longer_tips = [prop for prop in properties if prop.major_axis_length > threshold_length]

        # Create a contour image and highlight tips on the original frame
        contourImage = np.zeros_like(gray)
        highlightedTips = np.zeros_like(gray)
        tips_coordinates = []
        for prop in significantly_longer_tips:
            if prop.area >= minRegionSize:
                coords = prop.coords
                rr, cc = draw.polygon_perimeter(coords[:, 0], coords[:, 1], contourImage.shape)
                contourImage[rr, cc] = 1
                
                # Highlight the right tip of the ice crystal
                minr, minc, maxr, maxc = prop.bbox
                tip_right = (prop.centroid[0], maxc)
                tips_coordinates.append(tip_right)
                rr, cc = draw.disk((tip_right[0], tip_right[1]), radius=5, shape=highlightedTips.shape)
                highlightedTips[rr, cc] = 1

        # Overlay highlighted tips on the original frame
        color_highlightedTips = cv2.cvtColor(highlightedTips.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        frame_with_tips = cv2.addWeighted(frame, 1.0, color_highlightedTips, 0.5, 0)
        
        # Draw the predefined line
        cv2.line(frame_with_tips, (int(line_position_pixels), 0), (int(line_position_pixels), height), (0, 0, 255), 2)

        # Write the frame to the output video
        out.write(frame_with_tips)

    # Release everything if job is finished
    cap.release()
    out.release()

    print("Video processing complete. Output saved to:", output_path)

# Define the video path and the line position in microns
video_path = '/Users/kingchhum/Desktop/Lab Materials/Edge Detection/icecrystal_video.mp4'
output_path = '/Users/kingchhum/Desktop/Lab Materials/Edge Detection/icecrystal_highlighted.mp4'
line_position_microns = 400  # Adjust this value as needed
pixel_to_micron_conversion = 1  # Conversion factor obtained earlier

# Call the function to process the video
highlight_significantly_longer_tips_video(video_path, output_path, line_position_microns, pixel_to_micron_conversion)
