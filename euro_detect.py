import base64
from PIL import Image
from StringIO import StringIO
import numpy as np
import cv2
from flask import Flask, request
from flask_cors import CORS
import json
app = Flask(__name__)
CORS(app)

def readb64(base64_string):
    sbuf = StringIO()
    sbuf.write(base64.b64decode(base64_string))
    pimg = Image.open(sbuf)
    return np.array(pimg)
    #return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

@app.route("/", methods=['POST'])
def euro_detect():
    roi = readb64(request.data)
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    # Adaptive Thresholding
    gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
    thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)

    # Circle detection
    circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, 1, 64,
                                param1=20, param2=40, minRadius=24, maxRadius=96)

    cv2.imwrite('gray_blur.jpg', gray_blur)
    cv2.imwrite('thresh.jpg', thresh)
    print circles
    return json.dumps(circles.tolist(), ensure_ascii=False)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
