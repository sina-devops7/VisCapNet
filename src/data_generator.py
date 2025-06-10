# src/data_generator.py

import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


class ImageCaptionDataGenerator(Sequence):
    def __init__(self, image_features, captions_dict, tokenizer, max_length, vocab_size, batch_size=32):
        # dict: image_id -> feature vector
        self.image_features = image_features
        # dict: image_id -> [caption1, caption2, ...]
        self.captions_dict = captions_dict
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size

        self.image_ids = list(captions_dict.keys())
        self.indices = np.arange(len(self.image_ids))

    def __len__(self):
        return int(np.ceil(len(self.image_ids) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_image_ids = self.image_ids[start:end]

        X1, X2, y = [], [], []

        for img_id in batch_image_ids:
            feature = self.image_features[img_id]
            captions = self.captions_dict[img_id]

            for caption in captions:
                seq = self.tokenizer.texts_to_sequences([caption])[0]

                for i in range(1, len(seq)):
                    in_seq = seq[:i]
                    out_seq = seq[i]

                    in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                    out_seq = to_categorical(
                        [out_seq], num_classes=self.vocab_size)[0]

                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_seq)

        return [np.array(X1), np.array(X2)], np.array(y)
