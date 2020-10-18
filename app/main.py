from flask import Flask, Response

import logging
from app.camera import VideoCamera


logger = logging.getLogger()

logger.setLevel(logging.DEBUG)
app = Flask(__name__)


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, port='5000', host='0.0.0.0')
