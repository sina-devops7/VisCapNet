import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_glove_embeddings(glove_path, embedding_dim):
    """
    خواندن وکتورهای GloVe از فایل
    خروجی: dict کلمه به وکتور
    """
    embeddings_index = {}
    with open(glove_path, encoding='utf8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print(f'Found {len(embeddings_index)} word vectors.')
    return embeddings_index


def create_tokenizer(all_captions, num_words=10000):
    """
    آموزش Tokenizer روی کپشن‌ها
    """
    tokenizer = Tokenizer(
        num_words=num_words,
        lower=True,
        oov_token='<unk>',
        filters='',
    )
    tokenizer.fit_on_texts(all_captions)
    return tokenizer


def get_vocab_size(tokenizer):
    """
    گرفتن تعداد کل کلمات + 1 (معادل همون خط vocab_size = len(tokenizer.word_index) + 1)
    """
    return len(tokenizer.word_index) + 1


def get_max_length(all_captions):
    """
    گرفتن طول بلندترین کپشن (max_length)
    """
    return max(len(caption.split()) for caption in all_captions)


def create_embedding_matrix(tokenizer, embeddings_index, embedding_dim):
    """
    ساخت embedding_matrix مثل کد اصلی
    """
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None and i < vocab_size:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
