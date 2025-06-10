# src/feature_extractor.py

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input


def build_feature_extractor():
    base_model = EfficientNetB0(
        include_top=False, weights='imagenet', pooling='avg')
    return tf.keras.Model(inputs=base_model.input, outputs=base_model.output)


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img = np.array(img)
    img = preprocess_input(img)
    return img


def extract_features(image_dir, image_list, model):
    features = {}
    for image_id in tqdm(image_list):
        image_path = os.path.join(image_dir, image_id)
        img = load_and_preprocess_image(image_path)
        img = np.expand_dims(img, axis=0)
        feature = model.predict(img, verbose=0)
        features[image_id] = feature[0]
    return features


def save_features(features, output_path):
    import pickle
    with open(output_path, 'wb') as f:
        pickle.dump(features, f)
    print(f'✅ Features saved to: {output_path}')
