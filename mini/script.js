const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const emotionElement = document.getElementById('emotion');
const confidenceElement = document.getElementById('confidence');
const emotionCard = document.getElementById('emotion-card');
const emotionIcon = document.getElementById('emotion-icon');
const emotionInfo = document.getElementById('emotion-info');
const scoreBars = document.getElementById('score-bars');
const statusPill = document.getElementById('status-pill');
const statusText = document.getElementById('status-text');
const scanLine = document.getElementById('scan-line');
const faceLabel = document.getElementById('face-label');
const faceLabelText = document.getElementById('face-label-text');

const API_URL = 'http://127.0.0.1:5000/predict';
const CAPTURE_INTERVAL_MS = 700;
const SMOOTHING_WINDOW = 8; // average last N frames to reduce flicker/confusion

const scoreHistory = [];
let lastStableEmotion = null;

const EMOTION_META = {
    Angry: {
        icon: '😠',
        css: 'emotion-angry',
        title: 'Angry',
        description: 'Signals frustration, irritation, or strong displeasure — often from feeling blocked, treated unfairly, or under pressure.',
        cues: ['Furrowed brows', 'Tight lips or clenched jaw', 'Intense eye contact'],
        suggestion: 'Take a short pause, breathe slowly, and step away if the situation allows.',
    },
    Disgust: {
        icon: '🤢',
        css: 'emotion-disgust',
        title: 'Disgust',
        description: 'A reaction to something unpleasant, offensive, or morally uncomfortable.',
        cues: ['Wrinkled nose', 'Raised upper lip', 'Head pulled slightly back'],
        suggestion: 'Identify what triggered the reaction and whether it is a safety, hygiene, or values issue.',
    },
    Fear: {
        icon: '😨',
        css: 'emotion-fear',
        title: 'Fear',
        description: 'Indicates worry, anxiety, or sensing potential danger or uncertainty.',
        cues: ['Wide eyes', 'Raised eyebrows', 'Tense mouth or frozen expression'],
        suggestion: 'Ground yourself with slow breathing and focus on what is actually happening right now.',
    },
    Happiness: {
        icon: '😊',
        css: 'emotion-happiness',
        title: 'Happiness',
        description: 'Reflects positive mood — joy, satisfaction, amusement, or comfort.',
        cues: ['Smiling mouth', 'Raised cheeks', 'Bright, relaxed eyes'],
        suggestion: 'Good moment to engage socially, brainstorm, or continue tasks you enjoy.',
    },
    Sad: {
        icon: '😢',
        css: 'emotion-sad',
        title: 'Sad',
        description: 'Often linked to loss, disappointment, fatigue, or feeling low in energy.',
        cues: ['Downturned mouth', 'Drooping eyelids', 'Less expressive face'],
        suggestion: 'Rest, talk to someone you trust, or break tasks into smaller steps.',
    },
    Surprise: {
        icon: '😲',
        css: 'emotion-surprise',
        title: 'Surprise',
        description: 'A brief reaction to something unexpected — can be positive or negative.',
        cues: ['Raised eyebrows', 'Open eyes', 'Dropped jaw'],
        suggestion: 'Pause for a second to process the new information before reacting.',
    },
    Neutral: {
        icon: '😐',
        css: 'emotion-neutral',
        title: 'Neutral',
        description: 'A calm baseline state with no strong emotional signal detected.',
        cues: ['Relaxed features', 'Balanced mouth position', 'Steady gaze'],
        suggestion: 'Useful state for focused work, listening, or objective decision-making.',
    },
    Uncertain: {
        icon: '🤔',
        css: 'emotion-uncertain',
        title: 'Uncertain',
        description: 'The model saw a face but is not confident enough to pick one emotion clearly.',
        cues: ['Mixed expressions', 'Poor lighting', 'Face partially blocked (glasses, hair, angle)'],
        suggestion: 'Improve lighting, face the camera directly, and keep your full face visible.',
    },
    'No face detected': {
        icon: '👤',
        css: 'emotion-neutral',
        title: 'No Face Detected',
        description: 'The camera feed is active, but no face was found in the current frame.',
        cues: ['Face out of frame', 'Too dark', 'Camera blocked'],
        suggestion: 'Move closer, center your face, and ensure the room is well lit.',
    },
};

const ALL_EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happiness', 'Sad', 'Surprise', 'Neutral'];

let overlayFrame = null;
let isCapturing = false;

function setStatus(state, text) {
    statusPill.className = `status-pill status-${state}`;
    statusText.textContent = text;
}

function resizeOverlay() {
    overlay.width = video.clientWidth;
    overlay.height = video.clientHeight;
}

function clearOverlay() {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    faceLabel.classList.add('hidden');
    scanLine.classList.add('hidden');
}

function scaleBBox(bbox, frameWidth, frameHeight) {
    const scaleX = overlay.width / frameWidth;
    const scaleY = overlay.height / frameHeight;
    return {
        x: bbox.x * scaleX,
        y: bbox.y * scaleY,
        width: bbox.width * scaleX,
        height: bbox.height * scaleY,
    };
}

function drawFaceBox(bbox, emotion, confidence) {
    const time = Date.now() / 1000;
    const pulse = 0.5 + 0.5 * Math.sin(time * 4);
    const padding = 8 + pulse * 4;

    const x = bbox.x - padding;
    const y = bbox.y - padding;
    const w = bbox.width + padding * 2;
    const h = bbox.height + padding * 2;

    ctx.clearRect(0, 0, overlay.width, overlay.height);

    ctx.strokeStyle = `rgba(99, 102, 241, ${0.65 + pulse * 0.35})`;
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);

    ctx.strokeStyle = `rgba(129, 140, 248, ${0.25 + pulse * 0.25})`;
    ctx.lineWidth = 8;
    ctx.strokeRect(x, y, w, h);

    const corner = 18;
    ctx.strokeStyle = '#a5b4fc';
    ctx.lineWidth = 4;
    [
        [x, y + corner, x, y, x + corner, y],
        [x + w - corner, y, x + w, y, x + w, y + corner],
        [x, y + h - corner, x, y + h, x + corner, y + h],
        [x + w - corner, y + h, x + w, y + h, x + w, y + h - corner],
    ].forEach((segment) => {
        ctx.beginPath();
        ctx.moveTo(segment[0], segment[1]);
        ctx.lineTo(segment[2], segment[3]);
        ctx.lineTo(segment[4], segment[5]);
        ctx.stroke();
    });

    faceLabel.classList.remove('hidden');
    faceLabel.style.left = `${x}px`;
    faceLabel.style.top = `${Math.max(y - 8, 8)}px`;
    faceLabelText.textContent = `${emotion} · ${Math.round(confidence * 100)}%`;

    scanLine.classList.remove('hidden');
}

function animateOverlay(bbox, emotion, confidence) {
    if (overlayFrame) cancelAnimationFrame(overlayFrame);

    const draw = () => {
        drawFaceBox(bbox, emotion, confidence);
        overlayFrame = requestAnimationFrame(draw);
    };
    draw();
}

function renderEmotionInfo(emotion) {
    const meta = EMOTION_META[emotion] || EMOTION_META.Uncertain;
    emotionCard.className = `emotion-card ${meta.css}`;
    emotionIcon.textContent = meta.icon;
    emotionElement.textContent = meta.title;
    emotionInfo.innerHTML = `
        <h4>${meta.title}</h4>
        <p>${meta.description}</p>
        <p><strong>Common facial cues:</strong></p>
        <ul>${meta.cues.map((cue) => `<li>${cue}</li>`).join('')}</ul>
        <p><strong>Helpful tip:</strong> ${meta.suggestion}</p>
    `;
}

function renderScoreBars(scores, activeEmotion) {
    scoreBars.innerHTML = ALL_EMOTIONS.map((emotion) => {
        const value = scores?.[emotion] ?? 0;
        const percent = Math.round(value * 100);
        const activeClass = emotion === activeEmotion ? 'active' : '';
        return `
            <div class="score-row ${activeClass}">
                <span class="score-label">${emotion}</span>
                <div class="score-track">
                    <div class="score-fill" style="width: ${percent}%"></div>
                </div>
                <span class="score-value">${percent}%</span>
            </div>
        `;
    }).join('');
}

function smoothScores(newScores) {
    scoreHistory.push(newScores);
    if (scoreHistory.length > SMOOTHING_WINDOW) scoreHistory.shift();
    if (scoreHistory.length === 0) return {};

    const smoothed = {};
    ALL_EMOTIONS.forEach((emotion) => {
        const values = scoreHistory.map((s) => s[emotion] ?? 0);
        smoothed[emotion] = values.reduce((a, b) => a + b, 0) / values.length;
    });
    return smoothed;
}

function pickEmotion(smoothed) {
    const entries = ALL_EMOTIONS.map((e) => [e, smoothed[e] ?? 0]).sort((a, b) => b[1] - a[1]);
    const [topEmotion, topScore] = entries[0];
    const secondScore = entries[1][1];
    const margin = topScore - secondScore;

    if (topScore < 0.22 || margin < 0.05) {
        return { emotion: 'Uncertain', confidence: topScore, margin };
    }
    return { emotion: topEmotion, confidence: topScore, margin };
}

function handleNoFace(message = 'No face detected') {
    scoreHistory.length = 0;
    lastStableEmotion = null;
    clearOverlay();
    if (overlayFrame) {
        cancelAnimationFrame(overlayFrame);
        overlayFrame = null;
    }
    renderEmotionInfo(message);
    confidenceElement.textContent = 'Confidence: —';
    renderScoreBars({}, null);
}

async function startVideo() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: false,
        });
        video.srcObject = stream;
        video.addEventListener('loadedmetadata', resizeOverlay);
        window.addEventListener('resize', resizeOverlay);
        setStatus('live', 'Camera live');
    } catch (error) {
        console.error('Error accessing webcam:', error);
        setStatus('error', 'Camera blocked');
        emotionElement.textContent = 'Camera unavailable';
        emotionInfo.innerHTML = '<p>Please allow camera access in your browser and reload the page.</p>';
    }
}

async function captureFrame() {
    if (isCapturing || !video.videoWidth) return;
    isCapturing = true;

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);

    try {
        const imageBlob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.92));
        const formData = new FormData();
        formData.append('image', imageBlob);

        const response = await fetch(API_URL, { method: 'POST', body: formData });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Prediction failed');
        }

        if (!data.face_detected) {
            handleNoFace(data.emotion || 'No face detected');
            return;
        }

        const smoothed = smoothScores(data.scores);
        const result = pickEmotion(smoothed);
        lastStableEmotion = result.emotion;

        renderEmotionInfo(result.emotion);
        const marginPct = Math.round(result.margin * 100);
        confidenceElement.textContent =
            `Confidence: ${Math.round(result.confidence * 100)}% · Clarity: ${marginPct}%`;
        renderScoreBars(smoothed, result.emotion);

        const scaled = scaleBBox(data.bbox, data.frame_width, data.frame_height);
        animateOverlay(scaled, result.emotion, result.confidence);
        setStatus('live', `Analyzing (${data.detector || 'face'})`);
    } catch (error) {
        console.error('Error connecting to backend:', error);
        setStatus('error', 'Backend offline');
        handleNoFace('Server connection failed');
        emotionInfo.innerHTML = '<p>Make sure the Flask backend is running at <code>http://127.0.0.1:5000</code>.</p>';
    } finally {
        isCapturing = false;
    }
}

startVideo();
setInterval(captureFrame, CAPTURE_INTERVAL_MS);
