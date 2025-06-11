def train_test_split(mapping, train_ratio=0.9):
    """
    تقسیم آیدی تصاویر به train و test
    خروجی: train, test (لیست آیدی عکس)
    """
    image_ids = list(mapping.keys())
    split = int(len(image_ids) * train_ratio)
    train = image_ids[:split]
    test = image_ids[split:]
    return train, test
