from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
from keras.models import load_model

# Load the emotion detection model
model = load_model('emotion_model.h5')

# Define emotion labels (ensure these match your model)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happiness', 'Sad', 'Surprise', 'Neutral']

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Preprocess the image
def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None  # No face detected
    
    # Crop the largest detected face (assuming one face per image)
    x, y, w, h = max(faces, key=lambda bbox: bbox[2] * bbox[3])
    face = gray[y:y+h, x:x+w]

    # Resize to 48x48, normalize, and reshape for the model
    face = cv2.resize(face, (48, 48))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=(0, -1))  # Shape: (1, 48, 48, 1)
    return face

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict_emotion():
    try:
        # Get the image from the POST request
        file = request.files['image']
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Preprocess the image
        processed_frame = preprocess_image(frame)
        if processed_frame is None:
            return jsonify({'emotion': "No face detected"})

        # Predict emotion
        preds = model.predict(processed_frame)
        print("Raw Predictions:", preds)  # Debugging output

        # Identify emotion with highest confidence
        confidence = np.max(preds)
        if confidence > 0.4:  # Lower threshold if necessary
            emotion_index = np.argmax(preds, axis=1)[0]
            predicted_emotion = emotion_labels[emotion_index]
        else:
            predicted_emotion = "Uncertain"

        # Return result
        return jsonify({'emotion': predicted_emotion, 'confidence': float(confidence)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
