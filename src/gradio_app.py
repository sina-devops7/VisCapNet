import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

from src.caption_utils import predict_caption

# efficientnet مدل feature extractor را فقط یک بار بارگذاری کن
effnet = EfficientNetB0(include_top=False, weights="imagenet", pooling="avg")


def extract_features_pil(img_pil):
    img = img_pil.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    feature_vector = effnet.predict(img_array)
    return feature_vector


def generate_caption_gradio(
    image, model, tokenizer, max_length
):
    try:
        feature = extract_features_pil(image)
        predicted = predict_caption(model, feature, tokenizer, max_length)
        return f"🤖 **Predicted Caption:**\n{predicted}"
    except Exception as e:
        return f"خطا در پردازش تصویر: {str(e)}"


def launch_gradio_app(model, tokenizer, max_length):
    gr.Interface(
        fn=lambda image: generate_caption_gradio(
            image, model, tokenizer, max_length),
        inputs=gr.Image(type="pil"),
        outputs="text",
        title="Image Captioning - Gradio + EfficientNet",
        description="هر عکسی رو آپلود کن و کپشن تولید‌شده توسط مدل رو ببین!",
    ).launch(debug=True, share=True)
