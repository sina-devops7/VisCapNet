def save_model(model, path):
    model.save(path)


def load_model_(path):
    from keras.models import load_model
    return load_model(path)
