import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy import misc

# Load an example image
image = misc.face(gray=True)

# Apply Gaussian filter
filtered_image = gaussian_filter(image, sigma=5)

# Plot the original and filtered images
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Filtered Image')
plt.imshow(filtered_image, cmap='gray')
plt.show()
