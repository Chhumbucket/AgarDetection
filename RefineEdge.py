from skimage import io, filters, measure, morphology, draw
import matplotlib.pyplot as plt
import numpy as np

# Load the original image
image_path = '/Users/kingchhum/Desktop/Lab Materials/Edge Detection/icecrystal.png'
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

# Create a contour image and highlight tips on the original image
contourImage = np.zeros_like(image)
highlightedTips = np.zeros_like(image)
for prop in properties:
    if prop.area >= minRegionSize:
        coords = prop.coords
        rr, cc = draw.polygon_perimeter(coords[:, 0], coords[:, 1], contourImage.shape)
        contourImage[rr, cc] = 1
        
        # Highlight the tips of the ice crystals
        if prop.major_axis_length > 0:  # To ensure it is an elongated structure
            tips = coords[coords[:, 0] == coords[:, 0].max()]  # Bottom tip
            tips = np.vstack((tips, coords[coords[:, 0] == coords[:, 0].min()]))  # Top tip
            for tip in tips:
                rr, cc = draw.disk((tip[0], tip[1]), radius=5, shape=highlightedTips.shape)
                highlightedTips[rr, cc] = 1

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
ax[2].set_title('Highlighted Tips of Ice Crystals')
ax[2].axis('off')

plt.tight_layout()
plt.savefig("highlighted_tips.png")
plt.show()
