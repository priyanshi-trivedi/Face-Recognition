import cv2
import face_recognition
import pickle

with open("embeddings/embeddings.pkl", "rb") as f:
    knn_clf = pickle.load(f)

with open("last_recognized_name.txt", "r") as f:
    recognized_name = f.read().strip()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

print("[INFO] Webcam opened. Looking for a face")

last_recognized_name = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")

    if len(face_locations) > 0:
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for encoding in face_encodings:
            # ✅ Fix: Ensure n_neighbors does not exceed training samples
            if hasattr(knn_clf, "_fit_X") and len(knn_clf._fit_X) < knn_clf.n_neighbors:
                knn_clf.n_neighbors = len(knn_clf._fit_X)

            name = knn_clf.predict([encoding])[0]

            if name != last_recognized_name:
                print(f"[INFO] Recognized: {recognized_name}") 
                last_recognized_name = recognized_name

                cv2.putText(frame, f"Hello, {recognized_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imwrite(f"captured_face_{recognized_name}.jpg", frame)
                break  # Exit loop after first recognition

    else:
        cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Face Recognition", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break
    elif key == 13:  # Enter key to confirm
        print(f"[INFO] Captured image is {recognized_name}.")
        print(f"[INFO] Recognized: {recognized_name}")
        break

cap.release()
cv2.destroyAllWindows()
