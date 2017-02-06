import cv2
import sys

# Get user supplied values
imagePath = sys.argv[1]
cascPath = sys.argv[2]

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect coins in the image
coins = faceCascade.detectMultiScale(
    gray,
    scaleFactor=2.4,
    minNeighbors=58,
    minSize=(32, 32),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

print("Found {0} coins!".format(len(coins)))

# Draw a rectangle around the coins
for (x, y, w, h) in coins:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Coins found", image)
cv2.waitKey(0)
