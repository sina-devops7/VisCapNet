# src/text_processing.py

import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def create_tokenizer(captions_list, num_words=10000):
    tokenizer = Tokenizer(
        num_words=num_words,
        lower=True,
        oov_token='<unk>',
        filters=''
    )
    tokenizer.fit_on_texts(captions_list)
    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer, vocab_size


def get_max_length(captions_list):
    return max(len(caption.split()) for caption in captions_list)


def load_glove_embeddings(glove_path, embedding_dim):
    embeddings_index = {}
    with open(glove_path, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print(f'✅ Loaded {len(embeddings_index)} word vectors from GloVe.')
    return embeddings_index


def build_embedding_matrix(tokenizer, embeddings_index, vocab_size, embedding_dim):
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if i >= vocab_size:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
