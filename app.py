import os
import sqlite3
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import datetime

# --- Configuration ---
app = Flask(__name__)
DATABASE = 'database.db'
MODEL_PATH = 'face_emotionModel.h5'
# Define the emotions
EMOTION_MAP = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# --- Load Model and Face Detector ---
try:
    # Load the trained emotion detection model
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f" * Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f" * ERROR: Could not load model from {MODEL_PATH}")
    print(f" * {e}")
    print(" * Please run model_training.py first to generate the model file.")
    model = None

try:
    # Load the Haar Cascade for face detection from the cv2 library files
    # This is more reliable for deployment than a local file path
    haar_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    face_detector = cv2.CascadeClassifier(haar_cascade_path)
    if face_detector.empty():
        raise IOError(f"Could not load Haar cascade from {haar_cascade_path}")
    print(f" * Face detector loaded successfully.")
except Exception as e:
    print(f" * ERROR: Could not load face detector.")
    print(f" * {e}")
    face_detector = None


# --- Database Setup ---
def init_db():
    """Initializes the SQLite database and creates the 'predictions' table."""
    with app.app_context():
        db = sqlite3.connect(DATABASE)
        cursor = db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                detected_emotion TEXT NOT NULL,
                timestamp DATETIME NOT NULL
            )
        ''')
        db.commit()
        db.close()
    print(" * Database initialized.")


# --- Prediction Function ---
def predict_emotion(image_bytes):
    """
    Takes raw image bytes, detects a face, preprocesses it,
    and returns the predicted emotion.
    """
    if model is None or face_detector is None:
        return "Model or face detector not loaded"

    try:
        # Decode image from bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_detector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return "No face detected"

        # Use the first (and likely largest) face found
        (x, y, w, h) = faces[0]

        # Crop to the face region of interest (ROI)
        roi_gray = gray_img[y:y + h, x:x + w]

        # Resize to model's expected input size (48x48)
        roi_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Preprocess for model
        # 1. Convert to float
        img_pixels = tf.keras.preprocessing.image.img_to_array(roi_resized)
        # 2. Normalize
        img_pixels = img_pixels.astype('float32') / 255.0
        # 3. Add batch dimension (1, 48, 48, 1)
        # Note: img_to_array adds channel dim, so we just add batch dim
        img_pixels = np.expand_dims(img_pixels, axis=0)

        # Predict
        predictions = model.predict(img_pixels)

        # Get the emotion with the highest probability
        max_index = np.argmax(predictions[0])
        predicted_emotion = EMOTION_MAP[max_index]

        return predicted_emotion

    except Exception as e:
        print(f"Error during prediction: {e}")
        return f"Error: {e}"


# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    """Renders the main page with the upload form."""
    # Render the HTML template, passing no prediction result initially
    return render_template('index.html', prediction=None)


@app.route('/predict', methods=['POST'])
def upload_and_predict():
    """Handles the file upload, runs prediction, and logs to DB."""
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'image' not in request.files:
            return render_template('index.html', error="No file selected.")

        file = request.files['image']

        if file.filename == '':
            return render_template('index.html', error="No file selected.")

        if file:
            try:
                # Read image file bytes
                image_bytes = file.read()

                # Get the original filename
                filename = secure_filename(file.filename)

                # Get prediction
                emotion = predict_emotion(image_bytes)

                # Log to database if it's a valid emotion
                if emotion in EMOTION_MAP.values():
                    db = sqlite3.connect(DATABASE)
                    cursor = db.cursor()
                    timestamp = datetime.datetime.now()
                    cursor.execute(
                        "INSERT INTO predictions (filename, detected_emotion, timestamp) VALUES (?, ?, ?)",
                        (filename, emotion, timestamp)
                    )
                    db.commit()
                    db.close()

                # Render the page again, this time with the prediction result
                return render_template('index.html', prediction=f"Detected Emotion: {emotion}")

            except Exception as e:
                print(f"Error in /predict route: {e}")
                return render_template('index.html', error=f"An error occurred: {e}")

    # Fallback to redirect to home
    return redirect(url_for('index'))


# --- Run the App ---
if __name__ == "__main__":
    init_db()  # Create the database and table if they don't exist
    # For Render, gunicorn will run the 'app' object.
    # This 'app.run' is for local development only.
    app.run(debug=True, host='0.0.0.0', port=5000)