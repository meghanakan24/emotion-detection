const video = document.getElementById('video');
const emotionElement = document.getElementById('emotion');

// Initialize webcam feed
async function startVideo() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    } catch (error) {
        console.error('Error accessing webcam:', error);
        alert('Unable to access webcam. Please check permissions.');
    }
}

// Capture a frame and send it to the backend
async function captureFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageBlob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg'));
    const formData = new FormData();
    formData.append('image', imageBlob);

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData,
        });

        if (response.ok) {
            const data = await response.json();
            emotionElement.textContent = data.emotion;
        } else {
            emotionElement.textContent = 'Error in prediction.';
        }
    } catch (error) {
        console.error('Error connecting to backend:', error);
        emotionElement.textContent = 'Server connection failed.';
    }
}

// Start webcam feed and continuously capture frames
startVideo();
setInterval(captureFrame, 1000); // Capture frames every second
