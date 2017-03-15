import base64
from PIL import Image
from StringIO import StringIO
import numpy as np
from flask import Flask, request
from flask_cors import CORS
from euro_detect import euro_detect
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
def on_request():
    roi = readb64(request.data)
    circles = euro_detect(roi);
    return json.dumps(circles.tolist(), ensure_ascii=False)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
