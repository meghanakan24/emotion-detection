import cv2
import numpy as np
from keras.models import load_model
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained emotion recognition model
model = load_model('emotion_model.h5')

# Define the reduced emotion labels (5 emotions)
emotion_labels = ['Angry', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Preprocessing function
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        x, y, w, h = faces[0]  # Take the first detected face
        face = gray[y:y+h, x:x+w]
    else:
        return None  # No face detected

    # Resize to the required input size (48x48) and normalize
    face = cv2.resize(face, (48, 48))
    face = face.astype('float32') / 255  # Normalize to [0, 1]
    face = face.reshape(1, 48, 48, 1)  # Reshape for the model
    return face

# API endpoint for emotion prediction
@app.route('/predict', methods=['POST'])
def predict_emotion():
    try:
        # Receive the image file
        file = request.files['image']
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Preprocess the image
        processed_frame = preprocess_image(frame)
        if processed_frame is None:
            return jsonify({'emotion': "No face detected"}), 400

        # Predict the emotion
        preds = model.predict(processed_frame)
        
        # Debugging: Print raw predictions
        print("Raw Predictions:", preds)
        print("Emotion Confidences:", [f"{label}: {conf:.2f}" for label, conf in zip(emotion_labels, preds[0])])

        # Identify the emotion with the highest confidence
        emotion_index = np.argmax(preds, axis=1)[0]
        predicted_emotion = emotion_labels[emotion_index]
        
        return jsonify({'emotion': predicted_emotion})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': 'Something went wrong'}), 500

# Start the Flask server
if __name__ == '__main__':
    app.run(debug=True)
