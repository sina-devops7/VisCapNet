# src/evaluation.py

from src.caption_utils import predict_caption
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm.notebook import tqdm


def evaluate_bleu(model, features, tokenizer, mapping, test_keys, max_length):
    actual, predicted = [], []
    smoothie = SmoothingFunction().method4
    for key in tqdm(test_keys, desc="[INFO] Generating captions and evaluating BLEU"):
        # Get actual captions for the image
        captions = mapping[key]
        # Predict the caption for the image
        y_pred = predict_caption(model, features[key], tokenizer, max_length)
        # Prepare for BLEU calculation
        actual_captions = [caption.split() for caption in captions]
        y_pred = y_pred.split()
        actual.append(actual_captions)
        predicted.append(y_pred)
    # Calculate BLEU scores
    scores = {
        'BLEU-1': corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0), smoothing_function=smoothie),
        'BLEU-2': corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie),
        'BLEU-3': corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie),
        'BLEU-4': corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    }
    return scores
