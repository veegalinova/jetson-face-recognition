import pickle

import numpy as np
from pymongo import MongoClient
from bson.binary import Binary
import hnswlib


class DB:
    def __init__(self):
        self.connection = MongoClient(
            host="mongodb", port=27017, readPreference="secondaryPreferred"
        )
        self.database = self.connection["database"]

        self.people_collection = self.database["people"]
        self.logs_collection = self.database["logs"]

        self.dim = 128
        self.max_elements = 1000
        self.index, self.known_face_metadata = self.create_hnswlib_index()

    def create_hnswlib_index(self):
        index = hnswlib.Index(space="cosine", dim=self.dim)
        metadata = list(self.people_collection.find({}, {"_id": 0}))

        index.init_index(max_elements=self.max_elements)
        index.set_num_threads(4)
        index.set_ef(10)

        if metadata:
            existing_people = np.array([i["embedding"] for i in metadata])
            index.add_items(existing_people)

        return index, metadata

    def register_new_person(self, face_encoding, photo):
        new_person_id = len(self.index.get_ids_list())
        new_person_photo = Binary(pickle.dumps(photo, protocol=2))

        new_user_data = dict(id=new_person_id, embedding=list(face_encoding), photo=new_person_photo)
        self.people_collection.insert(new_user_data)
        self.known_face_metadata.append(
            dict(id=new_person_id, embedding=list(face_encoding))
        )

        self.index.add_items([face_encoding])
        return new_person_id

    def lookup_known_face(self, face_encoding, threshold_distance=0.08):
        if len(self.known_face_metadata) == 0:
            return None

        label, distance = self.index.knn_query(face_encoding, k=1)
        label = label[0][0]
        distance = distance[0][0]

        if distance < threshold_distance:
            user_data = self.known_face_metadata[label]
            user_data["distance"] = round(float(distance), 2)
            return user_data
        else:
            return None

    def update_photo(self, id, photo):
        photo = Binary(pickle.dumps(photo, protocol=2))
        self.people_collection.update_one(
            {"id": id, "updated": {"$exists": False}},
            {"$set": {"photo": photo, "updated": 1}},
            upsert=False,
        )

    def save_logs(self, data):
        self.logs_collection.insert(data)
