from src.data_generator import data_generator


def train_model(model, train_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size, epochs, callback):
    """
    آموزش مدل روی ژنراتور دقیقا مطابق کد اصلی
    """
    steps = len(train_keys) // batch_size
    model.fit(
        data_generator(train_keys, mapping, features, tokenizer,
                       max_length, vocab_size, batch_size),
        epochs=epochs,
        steps_per_epoch=steps,
        verbose=1,
        callbacks=[callback]
    )
