from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model


def build_captioning_model(
    vocab_size, max_length, embedding_matrix, embedding_dim=100, trainable_embed=True
):
    """
    ساخت شبکه عصبی کپشن‌زن همانند کد اصلی نوت‌بوک.
    """
    # image feature layers
    inputs1 = Input(shape=(1280,))
    fe1 = Dropout(0.4)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # sequence feature layers
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=max_length,
        trainable=trainable_embed,    # True یا False دقیقاً طبق کدت
        mask_zero=True
    )(inputs2)
    se2 = Dropout(0.4)(se1)
    se3 = LSTM(256)(se2)
    # decoder
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def visualize_model(model, filename):
    """
    ذخیره visualization ساختار مدل
    """
    plot_model(model, to_file=filename, show_shapes=True)
