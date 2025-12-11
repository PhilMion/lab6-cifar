import os
import sys

# --- ОПТИМІЗАЦІЯ ПАМ'ЯТІ ---
# Це критично важливо для Render Free Tier.
# Ми забороняємо TensorFlow шукати GPU, що економить ~200МБ RAM.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Прибираємо зайві логи TF

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# --- НАЛАШТУВАННЯ ---
# Ім'я файлу точно як на GitHub
MODEL_FILENAME = 'cnn_frog_bird_cat.h5'

# --- ДІАГНОСТИКА ---
print("--- ЗАПУСК СЕРВЕРА ---")
print(f"Робоча папка: {os.getcwd()}")
print("Файли в папці:", os.listdir(os.getcwd()))

# --- ЗАВАНТАЖЕННЯ МОДЕЛІ ---
MODEL_PATH = os.path.join(os.getcwd(), MODEL_FILENAME)
model = None

try:
    if os.path.exists(MODEL_PATH):
        print(f"Завантажую модель з: {MODEL_PATH} ...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("МОДЕЛЬ УСПІШНО ЗАВАНТАЖЕНА!")
    else:
        print(f"КРИТИЧНА ПОМИЛКА: Файл {MODEL_FILENAME} не знайдено!")
except Exception as e:
    print(f"КРИТИЧНА ПОМИЛКА при завантаженні моделі: {e}")

# Класи з твоєї лабораторної
CLASSES = {0: 'Bird (Птах)', 1: 'Cat (Кіт)', 2: 'Frog (Жаба)'}

def prepare_image(image_bytes):
    """Обробка зображення перед подачею в нейромережу"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize((32, 32)) # Розмір як при навчанні
        img_array = np.array(img) / 255.0 # Нормалізація
        
        # Якщо картинка має альфа-канал (прозорість), прибираємо його
        if img_array.shape[-1] == 4:
            img_array = img_array[..., :3]
            
        # Додаємо вимір batch (1, 32, 32, 3)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Помилка обробки зображення: {e}")
        return None

@app.route('/')
def home():
    status = "Model LOADED OK" if model else "Model NOT FOUND"
    return f"Status: {status}. Files: {str(os.listdir(os.getcwd()))}"

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded on server'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    try:
        file = request.files['file']
        img_array = prepare_image(file.read())
        
        if img_array is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Передбачення
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
        # Логуємо помилку на сервері
        print(f"Помилка під час predict: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Запуск
    app.run(debug=True, host='0.0.0.0', port=5000)
