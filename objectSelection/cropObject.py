import numpy as np
import cv2
img = cv2.imread('images/example_01.png', 0) # Read in your image
im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Your call to find the contours
print hierarchy
# idx = 0 # The index of the contour that surrounds your object
# mask = np.zeros_like(img) # Create mask where white is what we want, black otherwise
# cv2.drawContours(mask, [contours[idx]], 1, 255, -1) # Draw filled contour in mask
#
# out = np.zeros_like(img) # Extract out the object and place into output image
# out[mask == 255] = img[mask == 255]
# # cv2.imshow('Output', out)
# # cv2.waitKey(0)
#
# # Now crop
# (x, y) = np.where(mask == 255)
# (topx, topy) = (np.min(x), np.min(y))
# (bottomx, bottomy) = (np.max(x), np.max(y))
# out = out[topx:bottomx+1, topy:bottomy+1]
#
# # Show the output image
# cv2.imshow('Output', out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
