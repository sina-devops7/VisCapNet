# main.py

from src.gradio_app import launch_gradio_app
import os
import pickle
import numpy as np

from src.kaggle_download import download_flickr8k
from src.config import BASE_DIR, WORKING_DIR
from src.feature_extractor import (
    load_efficientnetb0,
    extract_features,
    save_features,
    load_features
)
from src.data_processing import (
    load_captions_file,
    build_image_caption_mapping,
    clean,
    build_all_captions_list
)
from src.tokenizer_prepare import (
    load_glove_embeddings,
    create_tokenizer,
    get_vocab_size,
    get_max_length,
    create_embedding_matrix
)
from src.dataset import train_test_split
from src.data_generator import data_generator
from src.model import build_captioning_model, visualize_model
from src.callbacks import CustomModelCheckpoint
from src.training import train_model
from src.caption_utils import predict_caption
from src.evaluation import evaluate_bleu
from keras.models import load_model

# ------------------------------------------------------------------------------------
# 1. Download dataset (optional, if not already available)
# print("Downloading Flickr8k dataset...")
# path = download_flickr8k()

# ------------------------------------------------------------------------------------
# 2. Feature extraction
features_path = os.path.join(WORKING_DIR, "features.pkl")
if not os.path.exists(features_path):
    print("[INFO] Extracting features from images...")
    model_effnet = load_efficientnetb0()
    features = extract_features(model_effnet, os.path.join(BASE_DIR, "Images"))
    save_features(features, features_path)
    print(f"[INFO] Feature extraction completed. Saved to {features_path}")
else:
    print(f"[INFO] Loading features from {features_path} ...")
    features = load_features(features_path)

# ------------------------------------------------------------------------------------
# 3. Captions processing
print("[INFO] Loading and processing captions...")
captions_doc = load_captions_file(os.path.join(BASE_DIR, "captions.txt"))
mapping = build_image_caption_mapping(captions_doc)
clean(mapping)
all_captions = build_all_captions_list(mapping)
print(
    f"[INFO] Total images: {len(mapping)} | Total cleaned captions: {len(all_captions)}")

# ------------------------------------------------------------------------------------
# 4. Load GloVe embeddings
print("[INFO] Loading GloVe embeddings...")
EMBEDDING_DIM = 100
glove_path = os.path.join(WORKING_DIR, "glove.6B.100d.txt")
embeddings_index = load_glove_embeddings(glove_path, EMBEDDING_DIM)
print(f"[INFO] Loaded GloVe vectors: {len(embeddings_index)}")

# ------------------------------------------------------------------------------------
# 5. Tokenizer and embedding matrix
print("[INFO] Creating tokenizer and embedding matrix...")
tokenizer = create_tokenizer(all_captions, num_words=10000)
vocab_size = get_vocab_size(tokenizer)
max_length = get_max_length(all_captions)
embedding_matrix = create_embedding_matrix(
    tokenizer, embeddings_index, EMBEDDING_DIM)
print(
    f"[INFO] Vocabulary size: {vocab_size} | Max caption length: {max_length}")

# ------------------------------------------------------------------------------------
# 6. Train/test split
print("[INFO] Splitting dataset (train/test)...")
train_keys, test_keys = train_test_split(mapping, train_ratio=0.9)
print(
    f"[INFO] Train images: {len(train_keys)} | Test images: {len(test_keys)}")

# ------------------------------------------------------------------------------------
# 7. Build model
print("[INFO] Building captioning model...")
model = build_captioning_model(
    vocab_size, max_length, embedding_matrix, EMBEDDING_DIM, trainable_embed=True
)
visualize_model(model, os.path.join(WORKING_DIR, 'model.png'))
print("[INFO] Model architecture saved as model.png")

# ------------------------------------------------------------------------------------
# 8. Training
EPOCHS = 10
BATCH_SIZE = 32
steps = len(train_keys) // BATCH_SIZE

checkpoint = CustomModelCheckpoint(
    save_path=os.path.join(WORKING_DIR, 'best_model.h5'),
    monitor='loss',
    save_best_only=True,
    save_freq=5
)

# Enable this block if you want to train:
if False:
    print("[INFO] Training model...")
    train_model(
        model, train_keys, mapping, features,
        tokenizer, max_length, vocab_size,
        BATCH_SIZE, EPOCHS, checkpoint
    )
    model.save(os.path.join(WORKING_DIR, 'best_model.keras'))
    print("[INFO] Training completed and best model saved.")

# ------------------------------------------------------------------------------------
# 9. Load trained model
model_path = os.path.join(WORKING_DIR, 'best_model.keras')
if os.path.exists(model_path):
    print(f"[INFO] Loading trained model from {model_path} ...")
    model = load_model(model_path)
else:
    print("[WARNING] Trained model not found. Using the current model.")

# ------------------------------------------------------------------------------------
# 10. Evaluate BLEU
print("[INFO] Evaluating BLEU scores on test set...")
bleu_scores = evaluate_bleu(
    model, features, tokenizer, mapping, test_keys, max_length
)
print("[RESULT] BLEU-1: {:.4f}".format(bleu_scores["BLEU-1"]))
print("[RESULT] BLEU-2: {:.4f}".format(bleu_scores["BLEU-2"]))
print("[RESULT] BLEU-3: {:.4f}".format(bleu_scores["BLEU-3"]))
print("[RESULT] BLEU-4: {:.4f}".format(bleu_scores["BLEU-4"]))

# ------------------------------------------------------------------------------------
# 11. Inference on a sample test image
sample_id = test_keys[0]
print("\n[INFO] Inference on one sample test image:")
print("Image ID:", sample_id)
print("Ground Truth Captions:", mapping[sample_id])
pred_caption = predict_caption(
    model, features[sample_id], tokenizer, max_length)
print("Predicted Caption:", pred_caption)

# ------------------------------------------------------------------------------------
# 12. Launch Gradio app
print("[INFO] Launching Gradio UI...")
launch_gradio_app(model, tokenizer, max_length)
