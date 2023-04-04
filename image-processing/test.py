import cv2

img = cv2.imread("image-processing/images/3419_joker.jpg")

cv2.imshow('new_image.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()