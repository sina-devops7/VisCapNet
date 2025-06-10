# src/caption_model.py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Add


def build_caption_model(vocab_size, max_length, embedding_dim=256, units=256):
    # Image feature input
    inputs_img = Input(shape=(1280,), name="image_features_input")
    img_dense = Dense(units, activation='relu')(inputs_img)
    img_dropout = Dropout(0.5)(img_dense)

    # Caption sequence input
    inputs_seq = Input(shape=(max_length,), name="text_sequence_input")
    seq_emb = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs_seq)
    seq_dropout = Dropout(0.5)(seq_emb)
    seq_lstm = LSTM(units)(seq_dropout)

    # Combine image and sequence
    combined = Add()([img_dropout, seq_lstm])
    decoder_output = Dense(units, activation='relu')(combined)
    outputs = Dense(vocab_size, activation='softmax')(decoder_output)

    model = Model(inputs=[inputs_img, inputs_seq], outputs=outputs)
    return model
