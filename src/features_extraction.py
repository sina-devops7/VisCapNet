# src/features_extraction.py

import os
import numpy as np
import pickle
from tqdm import tqdm
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model


def build_model():
    model = EfficientNetB0(weights="imagenet")
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    return model


def extract_features(directory, model, target_size=(224, 224)):
    features = {}
    for img_name in tqdm(os.listdir(directory)):
        img_path = os.path.join(directory, img_name)
        try:
            img = image.load_img(img_path, target_size=target_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            feature = model.predict(img_array, verbose=0)
            img_id = img_name.split('.')[0]
            features[img_id] = feature.squeeze()
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
    return features


def save_features(features, filename='features.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(features, f)


def load_features(filename='features.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)
