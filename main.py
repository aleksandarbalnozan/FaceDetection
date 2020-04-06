import cv2

video_capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('.\haarcascade_frontalface_default.xml')

while True:
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        frame, scaleFactor=1.5, minNeighbors=3)
    for (x, y, w, h) in faces:
        # if you want to capture croped image in grey/color image
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        cv2.imwrite('image.png', roi_gray)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), thickness=2)

    cv2.imshow('Camera', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
