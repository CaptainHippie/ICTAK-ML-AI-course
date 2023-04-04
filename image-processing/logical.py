import numpy as np

#### rectangel
import cv2

rectangle = np.zeros((300, 300), dtype = "uint8")

cv2.rectangle(rectangle, (25,25), (275, 275), 255, -1)

cv2.imwrite("image-processing/images/logical/rectangle-1.jpg", rectangle)


#### circle

circle = np.zeros((300, 300), dtype = "uint8")

cv2.circle(circle, (150,150), 150, 255, -1)

cv2.imwrite("image-processing/images/logical/circle-1.jpg", circle)

### bitwise and

bitand = cv2.bitwise_and(rectangle, circle)

cv2.imwrite("image-processing/images/logical/bitand.jpg", bitand)

### bitwise or

bitor = cv2.bitwise_or(rectangle, circle)

cv2.imwrite("image-processing/images/logical/bitor.jpg", bitor)

### bitwise xor
bitxor = cv2.bitwise_xor(rectangle, circle)

cv2.imwrite("image-processing/images/logical/bitxor.jpg", bitxor)