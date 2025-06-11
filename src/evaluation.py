from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def evaluate_bleu(model, features, tokenizer, mapping, test_keys, max_length):
    actual, predicted = list(), list()
    for key in test_keys:
        captions = mapping[key]
        y_pred = predict_caption(model, features[key], tokenizer, max_length)
        actual_captions = [caption.split() for caption in captions]
        y_pred = y_pred.split()
        actual.append(actual_captions)
        predicted.append(y_pred)
    smoothie = SmoothingFunction().method4
    bleu_scores = {
        "BLEU-1": corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0), smoothing_function=smoothie),
        "BLEU-2": corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie),
        "BLEU-3": corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie),
        "BLEU-4": corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie),
    }
    return bleu_scores
