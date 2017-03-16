import numpy as np
import cv2
import json
import math

FRAME_WIDTH = 640



class Coin:
    """Coin class for a euro coin."""

    def __init__(self, x, y, r):
        """Initializes the euro coin object."""
        self.x = x;
        self.y = y;
        self.r = r;
        self.results = {}
        self.feature = {}
        self.__process_color()
        self.__classification()

    def __process_color(self):
        """Process the colors of this coin's center area."""
        # Only use half of the coin area to determine it's center color
        r = self.r * 0.5

        roi = Coin.hsv[int(self.y-r):int(self.y+r), int(self.x-r):int(self.x+r)]

        if len(roi) > 0:
            self.feature['hue'] = (sum([pixel[0] for rows in roi for
                pixel in rows]) / len(roi) / len(roi[0]))
            self.feature['saturation'] = (sum([pixel[1] for rows in roi for
                pixel in rows]) / len(roi) / len(roi[0]))
            self.feature['lightness'] = (sum([pixel[2] for rows in roi for
                pixel in rows]) / len(roi) / len(roi[0]))


    def __classification(self):
        """Classify the coins using their colors."""
        for classification in Coin.classifier:
            self.results[classification] = {}
            for feature in Coin.classifier[classification]:
                f = Coin.classifier[classification][feature]
                x = self.feature[feature]

                # if x < f['minimum'] or x > f['maximum']:
                #     # No existing data matches
                #     self.results[classification][feature] = 99999
                #     continue
                # else:

                # Calculate the z-score of this feature
                z = (x - float(f['mean'])) / float(f['std_deviation'])
                self.results[classification][feature] = z

            z = self.results[classification]
            z = z['hue'] * 0.6 + z['saturation'] * 0.35 + z['lightness'] * 0.05

            # Convert the z-score to p value (probability)
            #p = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            p = 1.0 + math.erf(-z / math.sqrt(2.0))
            self.results[classification] = p



def euro_detect(rgb):
    """Detect the euro coins from a RGB stream of a colored image and returns
    the detection results."""

    # Assign Coin class variables
    Coin.rgb = rgb
    Coin.gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    Coin.hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    with open('euro_coin_detector_classifier.json') as infile:
        Coin.classifier = json.load(infile)['classification']

    # Adaptive Thresholding
    gray_blur = cv2.GaussianBlur(Coin.gray, (15, 15), 0)
    thresh = cv2.adaptiveThreshold(gray_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Circle detection
    circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, 1, 64,
                                param1=20, param2=40, minRadius=24,
                                maxRadius=96)

    coins = None
    if circles is not None:
        # Initialize coins using those detected circles
        coins = map(lambda c: Coin(c[0], c[1], c[2]), circles[0])


    # cv2.imwrite('gray_blur.jpg', gray_blur)
    # cv2.imwrite('thresh.jpg', thresh)

    return coins


def demo(roi):
    coins = euro_detect(roi)

    height, width, depth = roi.shape

    circle_mask = np.zeros((height, width), np.uint8)

    if coins is not None:

        for c in coins:
            # Draw into the circle mask
            cv2.circle(circle_mask, (int(c.x), int(c.y)), int(c.r),
                1, thickness=-1)

    masked_data = cv2.bitwise_and(roi, roi, mask=circle_mask)

    if coins is not None:
        for c in coins:
            # Draw the outer circle
            cv2.circle(roi, (int(c.x), int(c.y)), int(c.r),
                (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(roi, (int(c.x), int(c.y)), 2, (0, 0, 255), 3)

            # Draw the hue sampling area
            cv2.rectangle(masked_data, (int(c.x - c.r*0.5), int(c.y - c.r*0.5)),
                (int(c.x + c.r*0.5), int(c.y + c.r*0.5)), (0, 255, 0), 1)
            # Draw the descriptive text
            cv2.putText(masked_data, "Hue " + str(c.feature['hue']),
                (int(c.x - c.r), int(c.y - c.r - 32)),
                cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255), 1)
            cv2.putText(masked_data, "Saturation " +
                str(c.feature['saturation']),
                (int(c.x - c.r), int(c.y - c.r - 16)),
                cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255), 1)
            for w in sorted(c.results, key=c.results.get, reverse=True):
                cv2.putText(masked_data, w + " ("
                    + "{:.2f}".format(c.results[w] * 100) + "%)",
                    (int(c.x - c.r), int(c.y - c.r)),
                    cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255), 1)
                break


    cv2.imshow('Detected Coins', cv2.cvtColor(masked_data, cv2.COLOR_RGB2BGR))
    cv2.imshow('Video', cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    # If this script is running as a standalone program, start a demo
    option = raw_input('(V)ideo cam, or (L)oad an image file?')
    option = option.lower()
    if option == 'l':
        # Read from a file
        file_name = raw_input('Enter an image file name: ')

        bgr = cv2.imread(file_name)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        demo(rgb)
        while(True):
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    elif option == 'v':
        # Start the video camera and show the detection results in real-time.
        cap = cv2.VideoCapture(0)

        while(True):
            ret, frame = cap.read()
            height, width, depth = frame.shape
            roi = cv2.resize(frame, (FRAME_WIDTH, FRAME_WIDTH * height / width))
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            demo(rgb)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
