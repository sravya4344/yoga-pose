import os
import numpy as np
import imageio
from django.shortcuts import render, redirect
from django.core.files.storage import default_storage
from django.conf import settings
from django.contrib import messages

import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                    enable_segmentation=False, min_detection_confidence=0.5)

# Extract pose landmarks from video or gif
def extract_landmarks(video_path):
    try:
        reader = imageio.get_reader(video_path)
    except Exception as e:
        print(f"❌ Error reading {video_path}: {e}")
        return []

    landmarks_list = []
    for frame in reader:
        if frame.ndim == 2:
            frame = np.stack((frame,) * 3, axis=-1)
        elif frame.shape[2] == 4:
            frame = frame[:, :, :3]
        elif frame.shape[2] != 3:
            print(f"⚠️ Skipping frame with shape: {frame.shape}")
            continue

        frame_rgb = np.array(frame).astype(np.uint8)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            frame_landmarks = []
            for lm in results.pose_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])
            landmarks_list.append(frame_landmarks)

    reader.close()
    return landmarks_list

# Find average landmarks from dataset for that pose
def get_average_landmarks_for_asana(asana_name):
    dataset_path = os.path.join(settings.BASE_DIR, 'yoga', 'dataset')
    files = os.listdir(dataset_path)
    matching_files = [f for f in files if asana_name.lower() in f.lower()]

    all_avg = []
    for f in matching_files:
        full_path = os.path.join(dataset_path, f)
        landmarks = extract_landmarks(full_path)
        if landmarks:
            all_avg.append(np.mean(landmarks, axis=0))

    if all_avg:
        return np.mean(all_avg, axis=0)
    return None

# Compare uploaded video with dataset average
def compare_uploaded_video(uploaded_path, expected_avg):
    input_landmarks = extract_landmarks(uploaded_path)
    if not input_landmarks or expected_avg is None:
        return False
    input_avg = np.mean(input_landmarks, axis=0)
    distance = np.linalg.norm(input_avg - expected_avg)
    print("Distance:", distance)
    return distance < 0.1

# Django views
def home(request):
    return render(request, 'login.html')

def signup(request):
    if request.method == 'POST':
        return redirect('upload')
    return render(request, 'login.html')

def login_user(request):
    if request.method == 'POST':
        return redirect('upload')
    return render(request, 'login.html')

def upload_pose(request):
    if request.method == 'POST' and request.FILES.get('poseImage'):
        asana = request.POST.get('asana')
        file = request.FILES['poseImage']

        # Save uploaded file
        file_path = default_storage.save(file.name, file)
        full_path = os.path.join(settings.MEDIA_ROOT, file_path)

        # Get average landmarks from dataset
        expected_avg = get_average_landmarks_for_asana(asana)
        if expected_avg is None:
            messages.error(request, f"No training data found for {asana}.")
            return redirect('upload')

        is_correct = compare_uploaded_video(full_path, expected_avg)
        result = "✅ Correct Pose!" if is_correct else "❌ Incorrect Pose!"
        return render(request, 'result.html', {'asana': asana, 'result': result})

    return render(request, 'upload.html')

def result(request):
    return render(request, 'result.html')
