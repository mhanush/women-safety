import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model('cnn_model.h5')

# Define emotion labels (adjust these to match your dataset)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Extract the face ROI and resize it to 48x48
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))

        # Convert grayscale to RGB by stacking to create 3 channels
        face = np.stack((face,) * 3, axis=-1)
        
        # Normalize the pixel values and add batch dimension
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)

        # Make a prediction on the face
        preds = model.predict(face)[0]
        emotion_probability = np.max(preds)
        emotion_label = emotion_labels[preds.argmax()]

        # Display the emotion label and confidence on the frame
        cv2.putText(frame, f"{emotion_label}: {emotion_probability:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Emotion Detection", frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
