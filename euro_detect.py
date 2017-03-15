import numpy as np
import cv2

FRAME_WIDTH = 640

def euro_detect(rgb_stream):
    """Detect the euro coins from a RGB stream of a colored image and returns
    the detection results."""

    gray = cv2.cvtColor(rgb_stream, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb_stream, cv2.COLOR_RGB2HSV)

    # Adaptive Thresholding
    gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
    thresh = cv2.adaptiveThreshold(gray_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Circle detection
    circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, 1, 64,
                                param1=20, param2=40, minRadius=24,
                                maxRadius=96)



    # Color segmentation
    if circles is not None:
        circles = [circle[0] for circle in circles.tolist()]
        for i in range(len(circles)):
            c = circles[i]
            x, y, r = c[0], c[1], c[2]

            # Only use half of the coin area to determine it's center color
            r = r * 0.5

            roi = hsv[int(y-r):int(y+r), int(x-r):int(x+r)]
            #print roi
            hue_avg = sum([pixel[0] for rows in roi for pixel in rows]) / len(roi) / len(roi[0])

            circles[i].append(hue_avg)
            print circles[i]

    # cv2.imwrite('gray_blur.jpg', gray_blur)
    # cv2.imwrite('thresh.jpg', thresh)

    return circles


def demo(roi):
    circles = euro_detect(roi)

    height, width, depth = roi.shape

    circle_mask = np.zeros((height, width), np.uint8)

    if circles is not None:

        for c in circles:
            # Draw into the circle mask
            cv2.circle(circle_mask, (int(c[0]), int(c[1])), int(c[2]),
                1, thickness=-1)
    masked_data = cv2.bitwise_and(roi, roi, mask=circle_mask)

    if circles is not None:
        for c in circles:
            x, y, r = c[0], c[1], c[2]
            # Draw the outer circle
            cv2.circle(roi, (int(x), int(y)), int(r),
                (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(roi, (int(x), int(y)), 2, (0, 0, 255), 3)
            # Draw the hue sampling area
            cv2.rectangle(roi, (int(x - r*0.5), int(y - r*0.5)), (int(x + r*0.5), int(y + r*0.5)), (0, 255, 0), 1)
            # Draw the descriptive text
            cv2.putText(roi, "Hue " + str(c[3]), (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 2)


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
