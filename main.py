import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def preprocess_video(video_path, target_size):
    frames = []
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, target_size)
        frame = frame / 255.0

        frames.append(frame)

    cap.release()

    return np.array(frames)

# Step 3: Feature Extraction
def extract_features(frames):

    pass

# Step 4: Training Data Preparation
# Assuming you have a dataset of preprocessed videos and corresponding labels
def prepare_data(video_paths, labels, target_size):
    X = []
    y = []

    for video_path, label in zip(video_paths, labels):
        preprocessed_video = preprocess_video(video_path, target_size)
        features = extract_features(preprocessed_video)

        X.append(features)
        y.append(label)

    return np.concatenate(X), np.array(y)

# Step 5: Model Training
def train_model(X_train, y_train):
    svm = SVC()
    svm.fit(X_train, y_train)
    return svm

# Step 6: Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Step 7: Model Deployment
def deploy_model(model, video_path, target_size):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, target_size)
        frame = frame / 255.0

        features = extract_features(np.expand_dims(frame, axis=0))
        predicted_label = model.predict(features)

        cv2.putText(frame, predicted_label[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Step 1: Dataset Collection
video_paths = [r'C:\Users\hp\Desktop\VIDEO ANALYTICS\video1.mp4', r'C:\Users\hp\Desktop\VIDEO ANALYTICS\video2.mp4']
labels = ['dancing', 'cleaning']

# Step 2: Data Preprocessing
target_size = (224, 224)  # Target size for resizing frames

# Step 4: Training Data Preparation
X, y = prepare_data(video_paths, labels, target_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Training
model = train_model(X_train, y_train)

# Step 6: Model Evaluation
accuracy = evaluate_model(model, X_test, y_test)
print("Accuracy:", accuracy)

# Step 7: Model Deployment
video_path = r'C:\Users\hp\Desktop\VIDEO ANALYTICS\video.mp4'
deploy_model(model, video_path, target_size)
