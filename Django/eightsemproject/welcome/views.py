import cv2
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
import os
import base64
import json
from django.conf import settings
from django.http import JsonResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from .models import StudentInfo
import pandas as pd
from datetime import datetime
from openpyxl import load_workbook
from knnscratch import KNN
from multioutput import MultiOutput


# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('D:\\Python ML\\Automated Attendance System using Face Recognization\\Face-Anti Spoofing Detection_SCRATCH\\trained-model\\facedetection-model\\haarcascade_frontalface_default.xml')


# Load the trained model for liveness detection
model_path = 'D:\\Python ML\\Automated Attendance System using Face Recognization\\Face-Anti Spoofing Detection_SCRATCH\\Django\\eightsemproject\\static\models\\model.h5'
model = load_model(model_path)

def welcome(request):
    return render(request, 'welcome/welcome.html')

def register(request):
    return render(request, 'welcome/register.html')

def recordview(request):
    # Load data from the XLSX file
    df = pd.read_excel('D:\\Python ML\\Automated Attendance System using Face Recognization\\Face-Anti Spoofing Detection_SCRATCH\\Django\\eightsemproject\\attendance_records.xlsx')
    
    # Convert DataFrame to a list of lists
    data = df.values.tolist()
    # Add headers
    headers = df.columns.tolist()
    data.insert(0, headers)
    # print(data)
    # Find index of "Student-Semester"
    semester_index = headers.index('Student-Semester')
    # Send data and index to the template
    return render(request, 'welcome/record.html', {'data': data, 'semester_index': semester_index})

def extract_features(image_array):
    # Convert to grayscale if required
    gray_face = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    # Resize the image to a fixed size (if your feature extractor requires this)
    fixed_size = (128, 128)  # Example fixed size
    resized_face = cv2.resize(gray_face, fixed_size)
    # Feature extraction (e.g., using HOG descriptor)
    hog = cv2.HOGDescriptor()
    features = hog.compute(resized_face)
    # Convert features to a one-dimensional array
    feature_vector = features.flatten()
    return feature_vector


def preprocess_face(face):
    """
    Preprocess the face image for model prediction.
    """
    face_img = cv2.resize(face, (150, 150))  # Resize to match model input size
    face_img = keras_image.img_to_array(face_img)
    face_img = np.expand_dims(face_img, axis=0)
    face_img /= 255.0  # Normalize
    return face_img

def is_real_face(face_img):
    """
    Determine whether the given face image is real or spoof.
    """
    preprocessed_face = preprocess_face(face_img)
    prediction = model.predict(preprocessed_face)
    # return prediction[0] < 0.5  # Returns True if real, False if spoof
    if prediction[0] < 0.5:
        label = "Real"
    else:
        label = "Spoof"
    # print(label)
    return label


@csrf_exempt
def detect_faces(request):
    if request.method == 'POST':
        try:
            # Read the raw request body
            body = request.body.decode('utf-8')
            # Parse JSON data
            data = json.loads(body)
            frame_data = data.get('video_frame', '')
            if not frame_data:
                return JsonResponse({'status': 'error', 'message': 'No image data received'})
            # Decode the base64 image data
            image_data = base64.b64decode(frame_data)
            np_array = np.frombuffer(image_data, np.uint8)
            if np_array.size == 0:
                return JsonResponse({'status': 'error', 'message': 'Empty image data'})
            # Decode the image to BGR format
            frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            if frame is None:
                return JsonResponse({'status': 'error', 'message': 'Failed to decode image'})
            
            # Detect face start 
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face_img = frame[y:y + h, x:x + w]
            # Detect face END

            # real or spoof Start
                real_spoof_infer = is_real_face(face_img)
                # print(real_spoof_infer)
            # real or spoof end

            # comapre face with db data and fetch name and semester start 
            if real_spoof_infer == 'Real':
                face_vector = extract_features(face_img)
                # Load stored feature vectors and labels from the database
                students = StudentInfo.objects.all()
                feature_vectors = []
                labels = []
                semester =[]
                for student in students:
                    # Load the vector from binary data
                    vector = np.frombuffer(student.vector, dtype=np.float32)
                    feature_vectors.append(vector)
                    labels.append(student.name)
                    semester.append(student.semester)
                # Convert lists to numpy arrays
                feature_vectors = np.array(feature_vectors)
                labels = np.array(labels)
                semester = np.array(semester)
                combined_targets = np.column_stack((labels, semester))
                # Initialize and fit MultiOutput  KNN classifier scratch
                knn = KNN(k=1, distance_metric='euclidean', weights='uniform')  # You can change the number of neighbors
                multioutput_knn = MultiOutput(knn, k=1, distance_metric='euclidean', weights='uniform')
                multioutput_knn.fit(feature_vectors, combined_targets) # feature_vectors is X and combine_targets are Y
                predicted = multioutput_knn.predict([face_vector])[0]
                predicted_label = predicted[0]
                predicted_semester = predicted[1]
                # print("KNN RESULT: Label -", predicted_label, ", Semester -", predicted_semester)

                # make attendance with date for real student start
                # create a excel to store the a attendance of student
                current_date = datetime.now().strftime("%Y-%m-%d")
                attendance_data = {
                    "Attendance-Date": [current_date],
                    "Student-Name": [predicted_label],
                    "Student-Semester": [predicted_semester]
                    }
                # Convert attendance data to DataFrame
                attendance_df = pd.DataFrame(attendance_data)
                # Save DataFrame to Excel file
                excel_filename = "attendance_records.xlsx"
                try: 
                    # If the file exists, load it
                    existing_df = pd.read_excel(excel_filename)
                    # print(existing_df)
                    # Check if the record already exists
                    is_existing_record = (
                        (existing_df['Attendance-Date'] == current_date) &
                        (existing_df['Student-Name'] == predicted_label) &
                        (existing_df['Student-Semester'] == int(predicted_semester))
                    ).any()
                    # print(is_existing_record)
                    if not is_existing_record:
                        combined_df = pd.concat([existing_df, attendance_df], ignore_index=True)
                        combined_df.to_excel(excel_filename, index=False)
                        message = "Attendance Done"
                    else:
                        message = "Already done attendance"
                        # print('Already Attendance Done')
                except FileNotFoundError:
                    # If the file does not exist, create it
                    attendance_df.to_excel(excel_filename, index=False)
                    message = "Attendance Done"
                # make attendance with date for real student end

            else:
                message = "FOR ATTENDANCE REAL PERSON IS NEEDED"
            # comapre face with db data and fetch name END

            # Save the processed image start
            output_directory = os.path.join(settings.BASE_DIR, 'saved_image')
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            output_filename = 'detected_faces.jpg'
            output_path = os.path.join(output_directory, output_filename)
            cv2.imwrite(output_path, face_img)
            # Save the processed image END

            output_url = '/saved_image/detected_faces.jpg'

            return JsonResponse({'status': 'success', 'message':message, 'image_url': output_url,'liveliness':real_spoof_infer})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

@csrf_exempt
def storestudentsinfo(request):
    if request.method == 'POST':
        # Process the incoming data (example: extract name and image_data)
        name = request.POST.get('name')
        image_data = request.POST.get('image_data')
        semester = request.POST.get('semester')
        if not name or not image_data:
            return JsonResponse({'error': 'Name and image are required'}, status=400)
        # Decode the base64 image data
        format, imgstr = image_data.split(';base64,')
        img_data = base64.b64decode(imgstr)
        # Convert the decoded data to a numpy array and then to an OpenCV image
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Detect faces in the image start
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # Draw rectangles around the faces (optional, for visualization purposes)
        for (x, y, w, h) in faces:
            # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_img_store = img[y:y + h, x:x + w]
        # Detect faces in the image End
            
            # Save the processed image start
            output_directory = os.path.join(settings.BASE_DIR, 'saved_image')
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            output_filename = 'detected_faces_store.jpg'
            output_path = os.path.join(output_directory, output_filename)
            cv2.imwrite(output_path, face_img_store)
            # Save the processed image end
        
        # convert detected face to knn format vector_feature Start
        face_img_vector =extract_features(face_img_store)
        # Conver detected face to knn format vector_feature end

        # Store info in database start
        student = StudentInfo(name=name, image_data=image_data, vector=face_img_vector, semester=semester)
        student.save()
        # Store infor in database end
        
        # Your logic to store student info
        print("successfully stored info")
        return JsonResponse({'message': 'Information stored successfully'})
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)