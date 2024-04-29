from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# Загрузка сохраненной модели
model = load_model('my_model.h5')

# Загрузка изображения
img_path = 'img.png'
full_size_image = cv2.imread(img_path)
gray = cv2.cvtColor(full_size_image, cv2.COLOR_RGB2GRAY)

# Обнаружение лиц
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

labels = ['Anger', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

# Обработка каждого лица
for (x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray, (48, 48))
    img_pixels = image.img_to_array(roi_gray)
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    img_pixels /= 255

    # Предсказание эмоции
    predictions = model.predict(img_pixels)
    max_index = np.argmax(predictions[0])
    emotion = labels[max_index]

    # Отображение прямоугольника вокруг лица и текста с эмоцией
    cv2.rectangle(full_size_image, (x, y), (x+w, y+h), (0, 255, 0), 4)
    cv2.putText(full_size_image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 255, 0), 3)
    
# Создание окна отображения с нормализованным размером
cv2.namedWindow('Emotion', cv2.WINDOW_NORMAL)
# Изменение размера окна до 600x600
cv2.resizeWindow('Emotion', 600, 600)
# Отображение изображения
cv2.imshow('Emotion', full_size_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

