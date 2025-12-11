import os
import sys
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# --- НАСТРОЙКИ ---
MODEL_FILENAME = 'cnn_frog_bird_cat.h5'

# --- ДИАГНОСТИКА ФАЙЛОВ (ЭТО ПОМОЖЕТ НАЙТИ ОШИБКУ) ---
print("--- ЗАПУСК СЕРВЕРА ---")
print(f"Текущая папка: {os.getcwd()}")
print("Файлы в папке:", os.listdir(os.getcwd()))

# Проверяем полный путь
MODEL_PATH = os.path.join(os.getcwd(), MODEL_FILENAME)

model = None

if os.path.exists(MODEL_PATH):
    print(f"Файл модели найден: {MODEL_PATH}")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Модель успешно загружена!")
    except Exception as e:
        print(f"ОШИБКА загрузки модели: {e}")
else:
    print(f"ОШИБКА: Файл {MODEL_FILENAME} НЕ НАЙДЕН!")
    # Мы не останавливаем сервер, чтобы ты мог увидеть логи,
    # но предсказания работать не будут.

CLASSES = {0: 'Bird (Птах)', 1: 'Cat (Кіт)', 2: 'Frog (Жаба)'}

def prepare_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((32, 32))
    img_array = np.array(img) / 255.0
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    # Показываем статус на главной странице
    if model:
        return "Service is Running! Model loaded."
    else:
        files = str(os.listdir(os.getcwd()))
        return f"Service Running, BUT MODEL NOT FOUND. Files here: {files}"

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded on server'}), 500

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        img_array = prepare_image(file.read())
        prediction = model.predict(img_array)
        class_id = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))
        result = {
            'class_id': int(class_id),
            'class_name': CLASSES.get(class_id, "Unknown"),
            'confidence': confidence
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

