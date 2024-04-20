import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Загрузка сохраненной модели
model = load_model('my_model.h5')

# Загрузка каскада Хаара
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels = ['Anger', 'Contempt','Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

# Захват видео с веб-камеры
cap = cv2.VideoCapture(1)

while True:
    # Чтение кадра
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        emotion = labels[max_index]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Emotion', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


