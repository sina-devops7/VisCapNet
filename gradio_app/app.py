import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import pickle

# بارگذاری مدل EfficientNet برای استخراج ویژگی
effnet = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")

# بارگذاری مدل آموزش‌دیده و tokenizer و max_length
model = load_model("../weights/best_model.h5")

with open("../features/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("../features/max_length.pkl", "rb") as f:
    max_length = pickle.load(f)


def extract_features(image_pil):
    img = image_pil.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    feature_vector = effnet.predict(img_array)
    return feature_vector


def idx_to_word(integer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(image_feature):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        yhat = model.predict([image_feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text


def generate_caption_gradio(image):
    try:
        features = extract_features(image)
        caption = predict_caption(features)
        return caption
    except Exception as e:
        return f"خطا: {str(e)}"


iface = gr.Interface(
    fn=generate_caption_gradio,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Image Captioning با EfficientNet و LSTM",
    description="یک تصویر آپلود کن و کپشن تولید شده توسط مدل رو ببین!"
)

if __name__ == "__main__":
    iface.launch(share=True)
