from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import mediapipe as mp
from keras.models import load_model

model = load_model('emotion_model.h5')

# Standard FER2013 label order — must match training
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

app = Flask(__name__)
CORS(app)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
mp_face_detector = mp.solutions.face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.45,
)


def pad_bbox(x, y, w, h, frame_w, frame_h, padding=0.22):
    """Expand crop so eyebrows/mouth aren't clipped — critical for emotion accuracy."""
    pad_w = int(w * padding)
    pad_h = int(h * padding)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(frame_w, x + w + pad_w)
    y2 = min(frame_h, y + h + pad_h)
    return x1, y1, x2 - x1, y2 - y1


def detect_face_mediapipe(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_detector.process(rgb)
    if not results.detections:
        return None

    h, w = frame.shape[:2]
    best = max(
        results.detections,
        key=lambda d: d.location_data.relative_bounding_box.width
        * d.location_data.relative_bounding_box.height,
    )
    bb = best.location_data.relative_bounding_box
    x = int(bb.xmin * w)
    y = int(bb.ymin * h)
    bw = int(bb.width * w)
    bh = int(bb.height * h)
    x, y, bw, bh = pad_bbox(x, y, bw, bh, w, h)
    return (x, y, bw, bh)


def detect_face_haar(gray):
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=3,
        minSize=(48, 48),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda bbox: bbox[2] * bbox[3])
    return pad_bbox(int(x), int(y), int(w), int(h), gray.shape[1], gray.shape[0])


def detect_face(frame):
    bbox = detect_face_mediapipe(frame)
    if bbox is not None:
        return bbox, 'mediapipe'
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bbox = detect_face_haar(clahe.apply(gray))
    return bbox, 'haar'


def prepare_face_tensor(gray_face):
    """Build model input with contrast enhancement."""
    enhanced = clahe.apply(gray_face)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    resized = cv2.resize(enhanced, (48, 48))
    tensor = resized.astype('float32') / 255.0
    return np.expand_dims(tensor, axis=(0, -1))


def predict_face_tensor(tensor):
    return model.predict(tensor, verbose=0)[0]


def ensemble_predict(gray, bbox):
    """Average predictions across augmentations to reduce single-frame noise."""
    x, y, w, h = bbox
    face = gray[y : y + h, x : x + w]
    if face.size == 0:
        return None

    variants = [
        face,
        cv2.flip(face, 1),
    ]

    # Slight brightness variants help with lighting swings
    for alpha, beta in [(1.1, 10), (0.9, -5)]:
        adjusted = cv2.convertScaleAbs(face, alpha=alpha, beta=beta)
        variants.append(adjusted)

    preds = np.mean([predict_face_tensor(prepare_face_tensor(v)) for v in variants], axis=0)
    return preds


@app.route('/predict', methods=['POST'])
def predict_emotion():
    try:
        file = request.files['image']
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'face_detected': False, 'emotion': 'Invalid image'}), 400

        bbox, detector = detect_face(frame)
        if bbox is None:
            return jsonify({
                'face_detected': False,
                'emotion': 'No face detected',
                'confidence': 0,
                'scores': {},
            })

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        preds = ensemble_predict(gray, bbox)
        if preds is None:
            return jsonify({'face_detected': False, 'emotion': 'No face detected'})

        scores = {
            label: round(float(score), 3)
            for label, score in zip(emotion_labels, preds)
        }

        # Use margin between top-1 and top-2 as a quality signal
        sorted_preds = np.sort(preds)[::-1]
        confidence = float(sorted_preds[0])
        margin = float(sorted_preds[0] - sorted_preds[1])

        emotion_index = int(np.argmax(preds))
        predicted_emotion = emotion_labels[emotion_index]

        # Map Happy -> Happiness for frontend display consistency
        display_emotion = 'Happiness' if predicted_emotion == 'Happy' else predicted_emotion
        display_scores = dict(scores)
        if 'Happy' in display_scores:
            display_scores['Happiness'] = display_scores.pop('Happy')

        if confidence < 0.30 or margin < 0.06:
            display_emotion = 'Uncertain'

        x, y, w, h = bbox
        return jsonify({
            'face_detected': True,
            'emotion': display_emotion,
            'confidence': round(confidence, 3),
            'margin': round(margin, 3),
            'scores': display_scores,
            'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
            'frame_width': frame.shape[1],
            'frame_height': frame.shape[0],
            'detector': detector,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
