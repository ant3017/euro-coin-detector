import cv2
import numpy as np

img = cv2.imread('data/1.jpg')

#img = cv2.medianBlur(img,5)
gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

circles = cv2.HoughCircles(gimg, cv2.HOUGH_GRADIENT, 1, 64,
                            param1=60, param2=40, minRadius=24, maxRadius=96)

circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    # draw the outer circle
    cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imshow('detected circles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
