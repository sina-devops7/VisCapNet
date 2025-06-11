# src/kaggle_download.py
def download_flickr8k():
    import kagglehub
    # دانلود دیتاست Flickr8k از KaggleHub (عین نوت‌بوک)
    path = kagglehub.dataset_download("adityajn105/flickr8k")
    print("Path to dataset files:", path)
    return path
