import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Load the Haar cascade classifier for face detection
face_cascade_path = 'D:\\Python ML\\Automated Attendance System using Face Recognization\\Face-Anti Spoofing Detection_SCRATCH\\trained-model\\facedetection-model\\haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Load the trained model for liveness detection
model_path = 'D:\\Python ML\\Automated Attendance System using Face Recognization\\Face-Anti Spoofing Detection_SCRATCH\\trained-model\\liveliness-detection-model\\model.h5'
model = load_model(model_path)

# Open the laptop camera (default camera is usually index 0)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is captured correctly
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to grayscale (Haar cascades work better on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract face region from the frame
        face = frame[y:y + h, x:x + w]
        
        # Preprocess the face image for model prediction
        face_img = cv2.resize(face, (150, 150))  # Resize to match model input size
        face_img = keras_image.img_to_array(face_img)
        face_img = np.expand_dims(face_img, axis=0)
        face_img /= 255.0  # Normalize

        # Make prediction
        prediction = model.predict(face_img)

        # Determine the class and color
        if prediction[0] < 0.5:
            color = (255, 0, 255)  # Purple for "Real"
            label = "Real"
        else:
            color = (0, 0, 0)  # Black for "Spoof"
            label = "Spoof"

        # Draw rectangle around the detected face with the appropriate color
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the resulting frame
    cv2.imshow('Face Detection and Classification', frame)

    # Press 'q' to exit the video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
