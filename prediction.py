import cv2
import numpy as np
from google.colab.patches import cv2_imshow


# Load the image
img = cv2.imread('IMG__215325_fresh_leaf.png')
cv2_imshow(img)

# Convert the image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds of the green and yellow-green color
green_lower = np.array([60, 100, 100])
green_upper = np.array([80, 255, 255])

yellow_green_lower = np.array([30, 100, 100])
yellow_green_upper = np.array([60, 255, 255])

# Create a mask for the green and yellow-green color
green_mask = cv2.inRange(hsv, green_lower, green_upper)
yellow_green_mask = cv2.inRange(hsv, yellow_green_lower, yellow_green_upper)

# Count the number of pixels in the green and yellow-green mask
green_pixels = cv2.countNonZero(green_mask)
yellow_green_pixels = cv2.countNonZero(yellow_green_mask)

# Check which color is more dominant
if green_pixels > yellow_green_pixels:
    print("highly nutritious")
else:
    print("less nutritious")
