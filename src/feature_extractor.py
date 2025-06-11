import os
import pickle
from tqdm import tqdm
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


def load_efficientnetb0():
    """
    لود EfficientNetB0 با همان تنظیمات نوت‌بوک.
    """
    model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        pooling='avg'
    )
    model.summary()
    return model


def extract_features(model, images_dir):
    """
    استخراج feature همه تصاویر یک پوشه
    ورودی:
      - model : مدل EfficientNet
      - images_dir : مسیر پوشه تصاویر (مثلاً BASE_DIR + '/Images')
    خروجی:
      - dict از image_id -> feature (np array)
    """
    features = {}
    for img_name in tqdm(os.listdir(images_dir)):
        img_path = os.path.join(images_dir, img_name)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape(
            (1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = img_name.split('.')[0]
        features[image_id] = feature
    return features


def save_features(features, save_path):
    """
    ذخیره dict استخراج‌شده به pickle
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(features, f)


def load_features(load_path):
    """
    لود pickle فایل featureها
    """
    with open(load_path, 'rb') as f:
        features = pickle.load(f)
    return features
