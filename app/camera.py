import cv2

from app.recognition import Recognizer


class VideoCamera():
    def __init__(self):
        self.video = cv2.VideoCapture('/samples/Singapore.mp4')
        self.frame_counter = 0

    def __del__(self):
        self.video.release()

    def get_frame(self):
        status, image = self.video.read()
        self.frame_counter += 1
        if self.frame_counter == self.video.get(cv2.CAP_PROP_FRAME_COUNT):
            self.frame_counter = 0
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        bboxes, encodings, labels = Recognizer.recognize(image)
        proceseed_frame = self.draw_faces(image, bboxes, labels)

        return self.get_frame_bytes(proceseed_frame)

    @staticmethod
    def draw_faces(frame, bboxes, labels):
        for (top, right, bottom, left), label in zip(bboxes, labels):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        return frame

    @staticmethod
    def get_frame_bytes(frame):
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
