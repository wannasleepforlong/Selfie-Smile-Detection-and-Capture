import cv2
import os
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

window_position = (100, 100)

cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
cv2.moveWindow('Webcam', window_position[0], window_position[1])

cap = cv2.VideoCapture(0)

output_dir = 'Captures'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

interval = 5
last_capture = time.time()

while True:
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    curr = time.time()
    duration = curr - last_capture

    if duration >= interval:
        smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=20)
        for (x, y, w, h) in smiles:

            #cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            photo_name = os.path.join(output_dir, f'Smile_{int(curr)}.jpg')
            cv2.imwrite(photo_name, img)
            print(f"Smile detected! Photo saved: {photo_name}")

        last_capture = curr

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
