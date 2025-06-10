# src/caption_preprocessing.py

import os
import re
from tqdm import tqdm


def load_captions_file(path):
    with open(path, 'r') as f:
        next(f)  # skip the header
        captions_doc = f.read()
    return captions_doc


def create_mapping(captions_doc):
    mapping = {}
    for line in tqdm(captions_doc.strip().split('\n')):
        tokens = line.split(',')
        if len(tokens) < 2:
            continue
        image_id, caption = tokens[0], " ".join(tokens[1:])
        image_id = image_id.split('.')[0]
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption)
    return mapping


def clean_captions(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i].lower().strip()
            caption = re.sub(r'[^a-z ]', '', caption)
            caption = re.sub(r'\s+', ' ', caption)
            caption = f'startseq {caption.strip()} endseq'
            captions[i] = caption
    return mapping


def get_all_captions(mapping):
    all_captions = []
    for captions in mapping.values():
        all_captions.extend(captions)
    return all_captions
