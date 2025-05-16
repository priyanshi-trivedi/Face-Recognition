import face_recognition
import os
import numpy as np
import pickle
from sklearn import neighbors

dataset_dir = "dataset"
X, y = [], []

for person in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person)
    for image_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, image_name)
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            X.append(encodings[0])
            y.append(person)

knn_clf = neighbors.KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X, y)


with open("embeddings/embeddings.pkl", "wb") as f:
    pickle.dump(knn_clf, f)

print("Model trained and saved.")
