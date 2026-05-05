# Real-Time Face Recognition System

A Python-based application that performs real-time face detection and recognition using **OpenCV** for computer vision and **Scikit-Learn** for machine learning. This system allows you to build a custom face database and recognize faces through a live webcam feed.

## 🚀 Features
- **Data Collection**: Automatically captures and labels face samples from your webcam.
- **Preprocessing**: Handles grayscale conversion and image normalization (100x100 pixels).
- **KNN Classification**: Uses a K-Nearest Neighbors algorithm for high-speed recognition.
- **Unknown Detection**: Implements a distance threshold to identify and label "Unknown" individuals.
- **Confidence Scoring**: Displays a confidence percentage for each recognition.

## 📋 Prerequisites
- Python 3.x
- A webcam

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd <repository-folder>

2. **Install dependencies**:
The project requires specific versions of several libraries to ensure compatibility. Install them using the provided requirements.txt:
    ```bash
    pip install -r requirements.txt
    
## Project Structure

video_read.py: Script to capture and save face data for a specific person.
face_recog.py: Main script for real-time face recognition.
face_dataset/: Directory where captured .npy files are stored.
haarcascade_frontalface_alt.xml: Pre-trained Haar Cascade model for face detection.
 
## Usage
# Step 1: Collect Face Data  
  Run the data collection script to create a profile for yourself or others:

    ```bash
    python video_read.py
  -Enter the person's name when prompted.
  -The script will detect your face and save every 10th frame (to ensure data variety).
  -Press 'q' to stop capturing once you have enough samples (e.g., 20–50 samples).

# Step 2: Run Recognition
  Start the live recognition system:

    ```bash
    python face_recog.py
  -The script loads all .npy files from the face_dataset folder.
  -It trains the KNN model on the fly.
  -A video window will open showing blue boxes around detected faces with their names and confidence levels.
  -Press 'q' to exit.

## ⚙️ Technical Details
  -Classifier: KNeighborsClassifier with n_neighbors=5.
  -Thresholding: Faces with an average distance greater than 20,000 from the nearest neighbors are marked as "Unknown".
  -Detection: Utilizes the haarcascade_frontalface_alt.xml model for robust face localization.
