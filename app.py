from flask import Flask, render_template, Response, send_file, request, redirect, url_for, jsonify 
import cv2
import torch
import numpy as np
from datetime import datetime
import os
import pyttsx3
import threading

# === Constants ===
KNOWN_DISTANCE = 40  # cm
KNOWN_WIDTH = 14.3   # cm
PERSON_REAL_WIDTH = 30  # cm
FONTS = cv2.FONT_HERSHEY_SIMPLEX

# === Load YOLOv5 Model ===
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')


# === Haar Cascade ===
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# === Distance Calculation Functions ===
def focal_length(measured_distance, real_width, width_in_rf_image):
    return (width_in_rf_image * measured_distance) / real_width

def distance_finder(focal_len, real_width, width_in_frame):
    return (real_width * focal_len) / width_in_frame

def face_data(image):
    face_width = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    for (x, y, w, h) in faces:
        face_width = w
    return face_width

# === Focal Length Calculation from Reference Image ===
ref_image = cv2.imread("lena.png")
if ref_image is None:
    raise FileNotFoundError("Reference image not found.")
ref_face_width = face_data(ref_image)
if ref_face_width == 0:
    raise ValueError("No face detected in reference image.")
focal_length_found = focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, ref_face_width)

# === Flask Setup ===
app = Flask(__name__)
detection_log = []

# === Voice Engine Setup ===
engine = pyttsx3.init()
engine.setProperty('rate', 160)
spoken_objects = set()  # Track objects that have been spoken

def speak_object(label):
    def run():
        engine.say(f"{label} detected")
        engine.runAndWait()
    threading.Thread(target=run).start()

# === Video Capture ===
cap = cv2.VideoCapture(0)

def process_frame(frame):
    global detection_log, spoken_objects
    results = model(frame)
    detections = results.xyxy[0]
    total_objects = len(detections)

    for result in detections:
        x1, y1, x2, y2, conf, cls = map(int, result[:6])
        label = model.names[cls]
        confidence = float(result[4])

        if label not in spoken_objects:
            spoken_objects.add(label)
            speak_object(label)

        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        distance_label = label
        distance = None

        if label == 'person':
            box_width = x2 - x1
            if box_width > 0:
                distance = distance_finder(focal_length_found, PERSON_REAL_WIDTH, box_width)

            # Removed gender detection/classification here
            distance_label = "Person"

        if distance is not None:
            distance_text = f"Distance: {distance:.2f} cm"
            cv2.putText(frame, distance_text, (x1, y1 - 50), FONTS, 0.8, (0, 0, 255), 2)

        (tw, th), _ = cv2.getTextSize(distance_label, FONTS, 0.8, 2)
        label_bg_top_left = (x1, y1 - 30)
        label_bg_bottom_right = (x1 + tw + 10, y1 - 5)

        cv2.rectangle(frame, label_bg_top_left, label_bg_bottom_right, (0, 255, 255), -1)
        cv2.putText(frame, distance_label, (x1 + 5, y1 - 10), FONTS, 0.7, (0, 0, 0), 2)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        screenshot_name = f"{label}_{timestamp}.jpg"
        screenshot_path = os.path.join('static', 'screenshots', screenshot_name)
        cv2.imwrite(screenshot_path, frame)

        detection_log.append({
            "object": label,
            "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "confidence": round(confidence, 2),
            "distance": distance if distance is not None else "N/A",
            "screenshot": f"/static/screenshots/{screenshot_name}"
        })

    cv2.putText(frame, f"Total Objects Detected: {total_objects}", (10, 30), FONTS, 0.8, (0, 0, 255), 2)
    return frame

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/download_log')
def download_log():
    filename = 'detection_log.txt'
    with open(filename, 'w') as f:
        for log in detection_log:
            f.write(f"{log['time']} - {log['object']} - Confidence: {log['confidence']} - Distance: {log['distance']}\n")
    return send_file(filename, as_attachment=True)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    image = request.files['image']
    image_path = os.path.join('static', 'uploads', image.filename)
    image.save(image_path)
    img = cv2.imread(image_path)
    result_img = process_frame(img)
    result_path = os.path.join('static', 'uploads', 'result_' + image.filename)
    cv2.imwrite(result_path, result_img)
    return send_file(result_path, as_attachment=True)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    video = request.files['video']
    video_path = os.path.join('static', 'uploads', video.filename)
    video.save(video_path)
    cap = cv2.VideoCapture(video_path)
    out_path = os.path.join('static', 'uploads', 'result_' + video.filename)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(CV2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame)
        out.write(frame)

    cap.release()
    out.release()
    return send_file(out_path, as_attachment=True)

@app.route('/get_detection_data')
def get_detection_data():
    return jsonify(log=detection_log)

@app.route('/filter_log')
def filter_log():
    label_filter = request.args.get('label', '').lower()
    if label_filter:
        filtered_logs = [
            log for log in detection_log
            if label_filter in log['object'].lower()
        ]
    else:
        filtered_logs = detection_log
    return jsonify(filtered_logs)

@app.route('/admin')
def admin():
    return render_template('admin.html', detection_log=detection_log)

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/admin')
def admin_panel():
    # Ideally check for session here, skipping for demo
    return render_template('admin.html')

@app.route("/index")
def home():
    return render_template("index.html")  # or whatever your main page is

if __name__ == "__main__":
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    if not os.path.exists('static/screenshots'):
        os.makedirs('static/screenshots')
    app.run(debug=True)
