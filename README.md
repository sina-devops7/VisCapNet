# 🧠 Image Captioning with EfficientNet + LSTM + GloVe

This repository contains a deep learning pipeline for generating captions from images using:
- **EfficientNetB0** for feature extraction
- **LSTM** sequence modeling
- **GloVe word embeddings**
- **Gradio** for an interactive UI


<p align="center">
  <img src="https://s6.uupload.ir/files/d1989b39-5e67-493f-851d-4b50e29ef051_wx2w.png" width="350" alt="VisCapNet Project Hero"/>
</p>


---

## 📁 Dataset

We use the [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) dataset. Captions and images are loaded via `kagglehub`.


<p align="center">
  <img src="https://s6.uupload.ir/files/ai_captioning_in_a_high-tech_landscape_5nzk.png" width="550" alt="VisCapNet Hero Logo"/>
  <br><br>
  <b><big>VisCapNet: Deep Image Captioning with EfficientNet & LSTM</big></b>
  <br>
  <i>From Pixels to Poetry — Automatic Captioning on Flickr8k</i>
</p>

---

## 🌟 Overview

VisCapNet is an end-to-end deep learning pipeline that **automatically generates natural-language captions** for images. It leverages advanced computer vision (EfficientNet) and natural language processing (LSTM, GloVe) for a modern sequence-to-sequence architecture, built for research and reproducibility in Google Colab or any Python environment.

- **Frameworks:** TensorFlow / Keras, NLTK, Gradio, Colab-friendly.
- **Dataset:** [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- **Why?** Benchmark, prototype, or experiment with top-tier image-to-text systems as seen in CVPR/ACL papers.

---

## 🏗️ Model Architecture

<img src="https://s6.uupload.ir/files/model_dnxn.png" alt="Model Architecture" width="85%">

### 🚀 **Encoder–Decoder (Show and Tell Style)**

- **🖼️ Image Encoder:**
  - Features extracted from **EfficientNet** (1280-D vector, pretrained on ImageNet)
  - 🪁 Dropout
  - 🟩 Dense Layer (256-D) for semantic reduction

- **🔡 Caption Decoder:**
  - **Embedding** layer initialized with pretrained **GloVe-100d** vectors (semantic-rich input!)
  - 🪁 Dropout & **LSTM** (256 units) for temporal sequence modeling
  - Handles padded token sequences for variable-length captions

- **🔗 Fusion**
  - `add` merges image & text context → further Dense → **softmax** output over vocabulary

#### ✨ **Key features**
- 🏞️ Dual context (vision ⬌ language)
- 🔷 GloVe pretraining (faster convergence/semantics)
- 🪁 Strategic dropout throughout
- 🌐 Context fusion before prediction
- 🔮 Outputs word-by-word for rich, flexible captions

> ⚡️ *This design enables the model to literally “see” and “describe” using the best of both AI worlds.*

---

## 🚀 Quickstart

👨‍💻 **In Google Colab?**  
Just clone and run!  
```bash
git clone https://github.com/sina-devops7/VisCapNet.git
cd VisCapNet
```

### 1. Install & Dependencies
```bash
pip install -r requirements.txt
# Or, in Colab, use:
!pip install -q tensorflow gradio nltk kagglehub
```

### 2. Download Dataset & Glove
```python
import kagglehub
path = kagglehub.dataset_download("adityajn105/flickr8k")
print("[SUCCESS] Dataset ready at:", path)

!wget --no-check-certificate http://nlp.stanford.edu/data/glove.6B.zip
!unzip -o glove.6B.zip glove.6B.100d.txt -d /content
```

---

## 📊 Dataset Exploration & Visualization

```python
import matplotlib.pyplot as plt
import numpy as np
import os
def plot_sample_images(mapping, base_dir, n=4):
    np.random.seed(42)
    ids = np.random.choice(list(mapping.keys()), n, replace=False)
    fig, axes = plt.subplots(1, n, figsize=(16,6))
    for ax, image_id in zip(axes, ids):
        img_path = os.path.join(base_dir, 'Images', image_id + '.jpg')
        if not os.path.exists(img_path):
            ax.axis('off')
            continue
        img = plt.imread(img_path)
        ax.axis('off')
        captions = mapping[image_id]
        cap_text = "\n".join([f"{i+1}. {c[8:-6]}" for i, c in enumerate(captions[:3])])
        ax.set_title(cap_text, fontsize=9)
    plt.suptitle("Random Samples from Flickr8k with Reference Captions", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()
```

---

## 🏋️‍♂️ Training Pipeline

**Steps:**
1. **Feature Extraction:** EfficientNetB0, saved/cached in `.pkl`.
2. **Mapping & Cleaning:** Create image→captions map, preprocess/clean, add `startseq` & `endseq`.
3. **Tokenizer & Embedding Matrix:** Fit on corpus, build embedding matrix from GloVe vectors.
4. **Train/Test Split:** Standard 90/10 split on images.
5. **Custom Data Generator:** Efficiently yields (image, partial caption) → next-word pairs.
6. **Model Construction:** Encoder-Decoder as above.
7. **Custom Checkpointing:** Best model checkpointing each N epochs.
8. **Evaluation/Prediction:** BLEU scores, sample generations.
9. **Web UI:** Gradio app for upload & caption!

---

## 📈 Results

### 📊 **BLEU scores on test set:**  
`🌟 BLEU-1: 0.6198` &nbsp;&nbsp;|&nbsp;&nbsp; `✨ BLEU-2: 0.4389` &nbsp;&nbsp;|&nbsp;&nbsp; `🎯 BLEU-3: 0.3078` &nbsp;&nbsp;|&nbsp;&nbsp; `🏅 BLEU-4: 0.2121`

---

#### 🔍 **Interpretation**
- 📝 **BLEU-1** (~0.62): Good word overlap; model uses correct content words
- 🧩 **BLEU-4** (~0.21): Meaningful phrase reproduction, competitive for Flickr8k
- 🏞️ **Benchmarks**: BLEU-1 > 0.6 and BLEU-4 > 0.2 = Very Good

> These scores mean the model generates contextually accurate, fluent captions for natural images.  
> See paper benchmarks: [Karpathy & Fei-Fei 2015](https://arxiv.org/abs/1411.4555)

---

## 🤖 Live Demo — Gradio Web UI

Launch with:
```python
print("[INFO] Launching Gradio Web UI...")
launch_gradio_app(model, tokenizer, max_length)
```
- Upload any image & get a caption in seconds!
- Public URL auto-generated in Colab

---

## 🛠️ Advanced Features

- **LSTM decoder** with `mask_zero` for true padding
- **Custom callbacks** and model checkpointing logic
- **Extensible**: Plug in VGG, ResNet, or transformers with minimal code changes
- **Ready for Attention / Transformer upgrades**
- **Modular structure**: src/, notebooks/, utilities, ready for production or research expansion

---

## 🙋 Author

| 👤 Sina Mehrabadi | MSc AI/Robotics | KN Toosi University<br>GitHub: [sina-devops7](https://github.com/sina-devops7) |
|---|---|---|

---

## 📄 Citation

```bibtex
@misc{viscapnet2025,
  author = {Mehrabadi, Sina},
  title = {VisCapNet: A Deep Learning Pipeline for Image Captioning},
  howpublished = {\url{https://github.com/sina-devops7/VisCapNet}},
  year = {2025}
}
```

---

## 📝 License  
Apache 2.0 – Free for research, education, and open-source projects.  
See `LICENSE` for details.

---

<p align="center">
  <b>⭐️ Star the repo for future updates! PRs, issues, and feedback very welcome :)</b>
</p>
```

---

======
# VisCapNet
Deep learning-based image captioning system leveraging EfficientNet and GloVe embeddings to generate accurate captions for images.


