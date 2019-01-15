import cv2

image = cv2.imread('/home/palash/thesis/thesisML/objectSelection/images/sample_images/5_Color.png')
cv2.imshow('RealSense', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
image_bordered = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
cv2.imshow('RealSense', image_bordered)
cv2.waitKey(0)
cv2.destroyAllWindows()
image_bordered = cv2.copyMakeBorder(image_bordered, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0,0,255])
cv2.imshow('RealSense', image_bordered)
cv2.waitKey(0)
cv2.destroyAllWindows()