import re
import cv2
import os
import yaml
import requests
import numpy as np
import random
from PIL import Image
from io import BytesIO
from types import SimpleNamespace
from datetime import datetime, timedelta
import yt_dlp
import telegram


def load_config(path: str):
    with open(path, 'r') as fr:
        cfg = yaml.safe_load(fr)
        for k, v in cfg.items():
            if type(v) == dict:
                cfg[k] = SimpleNamespace(**v)
        cfg = SimpleNamespace(**cfg)
    return cfg


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    
def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb


def read_video(path, per_s=1):
    prev_frame_time = 0

    cap = cv2.VideoCapture(path)
    chunk = cap.get(cv2.CAP_PROP_FPS) * per_s
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
    
        if not ret:
            break
        else:
            # Display the resulting frame
            if count % round(chunk) == 0:
                frames.append(frame)
            count += 1

    return frames


def check_allowed_url(url):
    allowed_extensions_image = ['jpg', 'png', 'jpeg', 'webb']
    if url.split('.')[-1] in allowed_extensions_image:
        data_type = 21
        return data_type
    if 'www.youtube.com' in url:
        data_type=7
    else:
        data_type=20
    return data_type 


# # download image
# def download_image(image_url):
#     response = requests.get(image_url)
#     image = Image.open(BytesIO(response.content)).convert("RGB")
#     return image


def make_message(js):
    result = ''
    for k, v in js.items():
        result += '- {}: {} \n'.format(str(k), str(v))
    return result
    

# def call_api(url, link_api='http://172.18.5.44:8000/mlbigdata/vision/brandsafety/predict'):
#     response = requests.get(link_api + "?url=%s"%url)
#     result = response.json()
#     labels = result['label']
#     latency = result['infos']
#     return labels, latency