from skimage import io, filters, measure, morphology, draw
import matplotlib.pyplot as plt
import numpy as np

def highlight_significantly_longer_tips(image_path, line_position_microns, pixel_to_micron_conversion, std_factor=2):
    # Convert line position from microns to pixels
    line_position_pixels = line_position_microns * pixel_to_micron_conversion
    
    # Load the original image
    image = io.imread(image_path, as_gray=True)

    # Perform edge detection on the original image
    detectedEdges = filters.sobel(image)

    # Threshold the edge image to create a binary image
    binaryImage = detectedEdges > filters.threshold_otsu(detectedEdges)

    # Label the connected regions
    labeledImage, numRegions = measure.label(binaryImage, return_num=True, background=0)

    # Calculate the properties of the labeled regions
    properties = measure.regionprops(labeledImage)

    # Remove small regions which are considered as noise
    minRegionSize = 100  # You can adjust this value based on the noise size
    cleanImage = morphology.remove_small_objects(labeledImage, min_size=minRegionSize)

    # Calculate the average and standard deviation of the major axis lengths
    major_axis_lengths = [prop.major_axis_length for prop in properties if prop.area >= minRegionSize]
    mean_length = np.mean(major_axis_lengths)
    std_length = np.std(major_axis_lengths)
    threshold_length = mean_length + std_factor * std_length

    # Identify significantly longer tips
    significantly_longer_tips = [prop for prop in properties if prop.major_axis_length > threshold_length]

    # Create a contour image and highlight tips on the original image
    contourImage = np.zeros_like(image)
    highlightedTips = np.zeros_like(image)
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

    # Check if any tip crosses the predefined line
    feedback = any(tip[1] > line_position_pixels for tip in tips_coordinates)

    # Plot the results
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(contourImage, cmap='gray')
    ax[1].set_title('Contours after Noise Removal')
    ax[1].axis('off')

    ax[2].imshow(image, cmap='gray')
    ax[2].imshow(highlightedTips, cmap='jet', alpha=0.5)  # Overlay highlighted tips
    ax[2].axvline(x=line_position_pixels, color='red', linestyle='--')  # Draw the predefined line
    ax[2].set_title('Highlighted Significantly Longer Tips')
    ax[2].axis('off')

    plt.tight_layout()
    plt.savefig("highlighted_tips_with_feedback.png")
    plt.show()

    return feedback

# Define the image path and the line position in microns
image_path = '/Users/kingchhum/Desktop/Lab Materials/Edge Detection/icecrystal.png'
line_position_microns = 400  # Adjust this value as needed
pixel_to_micron_conversion = 1  # Conversion factor obtained earlier

# Call the function and get the feedback
feedback = highlight_significantly_longer_tips(image_path, line_position_microns, pixel_to_micron_conversion)
print("Feedback:", "Tip(s) passed the line." if feedback else "No tip passed the line.")