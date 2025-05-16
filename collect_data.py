import cv2
import os
name = input("Enter your name: ")
save_path = f"dataset/{name}"
os.makedirs(save_path, exist_ok=True)

with open("last_recognized_name.txt", "w") as f:
    f.write(name)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

print("[INFO] Webcam opened. Press Enter to capture a face")

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame.")
        break
    cv2.putText(frame, "Press Enter to capture the image", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Capturing Face", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 13:
       
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

      
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

       
        if len(faces) > 0:
            
            (x, y, w, h) = faces[0]

            face_img = frame[y:y + h, x:x + w]

            
            count += 1
            cv2.imwrite(f"{save_path}/{count}.jpg", face_img)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            print(f"[INFO] Face captured and saved to {save_path}/{count}.jpg")
            break  


    if key == 27:
        print("[INFO] ESC pressed, stopping the capture.")
        break

cap.release()
cv2.destroyAllWindows()
