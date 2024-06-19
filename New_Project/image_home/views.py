import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from django.shortcuts import render
from .forms import ImageUploadForm
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load pre-trained VGG16 model + higher level layers
model = VGG16(weights='imagenet', include_top=False)

# Cache for base image features
base_image_features_cache = {}

def extract_features(image_path):
    # Read and preprocess the image
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (224, 224))  # VGG16 input size
    img_array = np.expand_dims(img_resized, axis=0)
    img_preprocessed = preprocess_input(img_array)

    # Extract features
    features = model.predict(img_preprocessed)
    return features.flatten()

def compare_images(features1, features2):
    # Compute cosine similarity
    similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
    return similarity * 100  # Convert to percentage

def handle_uploaded_image(file):
    file_path = os.path.join('static/images', file.name)
    with open(file_path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    return file_path

def process_image(file, base_images, threshold):
    uploaded_image_url = handle_uploaded_image(file)
    uploaded_features = extract_features(uploaded_image_url)

    mse_list = {}
    for base_image in base_images:
        base_image_path = os.path.join('static/Base_Images', base_image)
        if base_image not in base_image_features_cache:
            base_image_features_cache[base_image] = extract_features(base_image_path)
        matching_percentage = compare_images(base_image_features_cache[base_image], uploaded_features)
        mse_list[base_image_path] = matching_percentage

    most_matching_image_url = None
    most_matching_percentage = 0
    for image, matching_percentage in mse_list.items():
        if matching_percentage > threshold and matching_percentage > most_matching_percentage:
            most_matching_percentage = matching_percentage
            most_matching_image_url = image

    return {
        'uploaded_image_url': uploaded_image_url.replace('static/', ''),
        'most_matching_image_url': most_matching_image_url.replace('static/', '') if most_matching_image_url else None,
        'most_matching_percentage': most_matching_percentage if most_matching_image_url else None
    }

def view_home(request):
    results = []
    threshold = 75

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        threshold = float(request.POST.get('threshold', 75))
        base_images = os.listdir('static/Base_Images')

        if form.is_valid():
            images = request.FILES.getlist('images')
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_image, file, base_images, threshold) for file in images]
                results = [future.result() for future in futures]

    else:
        form = ImageUploadForm()

    context = {
        'form': form,
        'results': results,
        'threshold': threshold
    }

    return render(request, 'home/index.html', context)
