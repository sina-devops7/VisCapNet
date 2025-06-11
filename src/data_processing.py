import os
import re
from tqdm import tqdm


def load_captions_file(captions_path):
    """
    لود captions.txt و گرفتن string کل فایل (skip first line).
    """
    with open(captions_path, 'r') as f:
        next(f)
        captions_doc = f.read()
    return captions_doc


def build_image_caption_mapping(captions_doc):
    """
    ساخت dict mapping: image_id -> list of captions
    """
    mapping = {}
    for line in tqdm(captions_doc.strip().split('\n')):
        tokens = line.split(',')
        if len(line) < 2:
            continue
        image_id, caption = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        caption = " ".join(caption)
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption)
    return mapping


def clean(mapping):
    """
    همان تابع clean نوت‌بوک (کوچک‌سازی، حذف کاراکتر غیرحروفی، حذف فاصله اضافی، افزودن startseq/endseq)
    """
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            caption = caption.lower().strip()
            caption = re.sub(r'[^a-z ]', '', caption)
            caption = re.sub(r'\s+', ' ', caption)
            # . . . حذف stopwords را قرار ندادم چون توی کدت غیرفعال بود
            caption = f'startseq {caption.strip()} endseq'
            captions[i] = caption
    # تغییر روی mapping به صورت in-place است


def build_all_captions_list(mapping):
    """
    تولید یک لیست از همه کپشن‌ها (پیش‌پردازش‌شده)
    """
    all_captions = []
    for key in mapping:
        for caption in mapping[key]:
            all_captions.append(caption)
    return all_captions
