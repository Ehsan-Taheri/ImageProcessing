import cv2 as cv
import numpy as np

# Load the cascade classifiers
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

# Check if the cascade files were loaded correctly
if face_cascade.empty() or eye_cascade.empty():
    raise IOError("Error loading cascade files")

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        frame_roi = frame[y:y+h, x:x+w]
        frame_gray_roi = frame_gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(frame_gray_roi)
        for(ex, ey, ew, eh) in eyes:
            cv.rectangle(frame_roi, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    cv.imshow('frame', frame)

    exit_key = cv.waitKey(5) & 0xFF
    if exit_key == 27:  
        break

cv.destroyAllWindows()
cap.release()