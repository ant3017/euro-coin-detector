import numpy as np
import cv2

FRAME_WIDTH = 640

def euro_detect(rgb_stream):
    gray = cv2.cvtColor(rgb_stream, cv2.COLOR_RGB2GRAY)

    # Adaptive Thresholding
    gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
    thresh = cv2.adaptiveThreshold(gray_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Circle detection
    circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, 1, 64,
                                param1=20, param2=40, minRadius=24,
                                maxRadius=96)

    # cv2.imwrite('gray_blur.jpg', gray_blur)
    # cv2.imwrite('thresh.jpg', thresh)

    return circles



if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while(True):
        ret, frame = cap.read()
        height, width, depth = frame.shape
        roi = cv2.resize(frame, (FRAME_WIDTH, FRAME_WIDTH * height / width))

        circles = euro_detect(roi)

        circle_mask = np.zeros((height, width), np.uint8)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))

            for i in circles[0, :]:
                # Draw into the circle mask
                cv2.circle(circle_mask, (i[0], i[1]), i[2], 1, thickness=-1)
        masked_data = cv2.bitwise_and(roi, roi, mask=circle_mask)

        if circles is not None:
            for i in circles[0, :]:
                # Draw the outer circle
                cv2.circle(roi, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Draw the center of the circle
                cv2.circle(roi, (i[0], i[1]), 2, (0, 0, 255), 3)
                # Draw the descriptive text
                cv2.putText(roi, "Euro Coin", (i[0], i[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 2)


        cv2.imshow('Detected Coins', masked_data)
        cv2.imshow('Video', roi)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
