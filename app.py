import os
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Ініціалізація Flask додатку
app = Flask(__name__)

# Завантаження моделі (робимо це один раз при запуску)
# Переконайтеся, що файл моделі лежить поруч з app.py
MODEL_PATH = 'cifar10_subset_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Словник класів (з твоєї 5-ї лаби: 2->Bird, 3->Cat, 6->Frog)
# Але в моделі вони перекодовані в 0, 1, 2
CLASSES = {0: 'Bird (Птах)', 1: 'Cat (Кіт)', 2: 'Frog (Жаба)'}

def prepare_image(image_bytes):
    """Функція для обробки зображення перед подачею в модель"""
    # Відкриваємо зображення з байтів
    img = Image.open(io.BytesIO(image_bytes))
    
    # Змінюємо розмір до 32x32 (як у тренувальних даних)
    img = img.resize((32, 32))
    
    # Конвертуємо в масив та нормалізуємо (ділимо на 255)
    img_array = np.array(img) / 255.0
    
    # Якщо зображення має 4 канали (RGBA), прибираємо прозорість
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
        
    # Додаємо вимір batch_size (модель чекає (1, 32, 32, 3))
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/')
def home():
    return "ML Model Service is Running! Send a POST request to /predict"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Перевіряємо, чи є файл у запиті
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        # Обробка зображення
        img_array = prepare_image(file.read())
        
        # Прогноз
        prediction = model.predict(img_array)
        class_id = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))
        
        # Формування відповіді
        result = {
            'class_id': int(class_id),
            'class_name': CLASSES[class_id],
            'confidence': confidence
        }
        
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Запуск сервера локально
    app.run(debug=True, host='0.0.0.0', port=5000)