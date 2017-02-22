import numpy as np
import cv2

FRAME_WIDTH = 640

def run_main():
    cap = cv2.VideoCapture(0)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while(True):
        ret, frame = cap.read()
        height, width, depth = frame.shape
        roi = cv2.resize(frame, (FRAME_WIDTH, FRAME_WIDTH * height / width))
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
        thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 1)


        # Circle detection
        circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, 1, 64,
                                    param1=60, param2=40, minRadius=24, maxRadius=96)


        circle_img = np.zeros((height, width), np.uint8)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # draw the outer circle
                cv2.circle(roi, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # draw the center of the circle
                cv2.circle(roi, (i[0], i[1]), 2, (0, 0, 255), 3)

                cv2.circle(circle_img, (i[0], i[1]), i[2], 1, thickness=-1)


        masked_data = cv2.bitwise_and(roi, roi, mask=circle_img)



        cv2.imshow("Adaptive Thresholding", thresh)
        cv2.imshow('Detected Coins', masked_data)
        cv2.imshow('Video', roi)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_main()
