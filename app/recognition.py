from datetime import datetime, timedelta

import face_recognition

from app import db, LAST_SAVE


class Recognizer:
    @staticmethod
    def recognize(frame):
        global LAST_SAVE
        # BGR -> RGB
        rgb_frame = frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model='cnn')

        face_labels = []
        names = []
        for face_location, face_encoding in zip(face_locations, face_encodings):
            user_data = db.lookup_known_face(face_encoding)
            top, right, bottom, left = face_location
            photo = frame[top:bottom, left:right]
            if user_data is not None:
                face_label = "{0}".format(user_data['id'])
                names.append(user_data['id'])
                db.update_photo(user_data['id'], photo)
            else:
                id = db.register_new_person(face_encoding, photo)
                face_label = "New: {0}".format(id)
                names.append(id)

            face_labels.append(face_label)

        if datetime.now() - LAST_SAVE > timedelta(seconds=10):
            for label in names:
                db.save_logs(dict(label=label, time=datetime.now()))
            LAST_SAVE = datetime.now()

        return face_locations, face_encodings, face_labels
