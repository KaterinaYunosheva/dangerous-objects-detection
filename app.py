from flask import Flask, request, render_template, send_from_directory, redirect, url_for
from ultralytics import YOLO
import cv2
import os
import uuid
import sqlite3
from datetime import datetime
import pandas as pd
from reportlab.pdfgen import canvas

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Загрузка модели YOLOv8
model = YOLO('yolov8n.pt')

# Подключение к базе данных
def init_db():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            object_class TEXT,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Сохранение результата в БД
def save_to_db(object_class, confidence):
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO detections (id, timestamp, object_class, confidence)
        VALUES (?, ?, ?, ?)
    ''', (str(uuid.uuid4()), datetime.now().isoformat(), object_class, confidence))
    conn.commit()
    conn.close()

# Обработка изображения
def process_image(path):
    results = model.predict(source=path, conf=0.5, classes=[0, 1])  # Классы: нож (0), пистолет (1)
    annotated = results[0].plot()
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], f'result_{os.path.basename(path)}')
    cv2.imwrite(output_path, annotated)
    return output_path, results

# Обработка видео
def process_video(path):
    cap = cv2.VideoCapture(path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], f'result_{os.path.basename(path)}')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, conf=0.5, classes=[0, 1])
        annotated = results[0].plot()
        out.write(annotated)
    cap.release()
    out.release()
    return output_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            ext = os.path.splitext(file.filename)[1].lower()
            filename = f"{uuid.uuid4()}{ext}"
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            if ext in ['.mp4', '.avi']:
                processed_path = process_video(path)
            else:
                processed_path, results = process_image(path)
                for r in results:
                    for cls, conf in zip(r.boxes.cls, r.boxes.conf):
                        save_to_db(model.names[int(cls)], float(conf))
            return redirect(url_for('results', filename=os.path.basename(processed_path)))
    return render_template('index.html')

@app.route('/results/<filename>')
def results(filename):
    return render_template('results.html', filename=filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/history')
def history():
    conn = sqlite3.connect('history.db')
    df = pd.read_sql('SELECT * FROM detections', conn)
    conn.close()
    return render_template('history.html', data=df.to_dict(orient='records'))

@app.route('/generate_report')
def generate_report():
    conn = sqlite3.connect('history.db')
    df = pd.read_sql('SELECT * FROM detections', conn)
    conn.close()
    df.to_excel('report.xlsx', index=False)
    return send_from_directory('.', 'report.xlsx')

if __name__ == '__main__':
    app.run(debug=True)
