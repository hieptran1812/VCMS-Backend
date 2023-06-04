import os
import unidecode
import string
import yt_dlp
import requests
from PIL import Image
from io import BytesIO


def check_allowed_url(url):
    allowed_extensions_image = ['jpg', 'png', 'jpeg', 'webb']
    if url.split('.')[-1] in allowed_extensions_image:
        data_type = 21
        return data_type
    if 'youtu' in url:
        data_type=7
    else:
        data_type=20
    return data_type 


def download_image(image_url):
    img_data = requests.get(image_url).content
    img_path = '../upload/image_name.jpg'
    with open(img_path, 'wb') as handler:
        handler.write(img_data)
    return img_path


def check_video(video_url):
    ydl_opts = {}
    check = 1
    error_detail = None
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(video_url, download=False)
        if result['duration'] > 30*60:
            check = 0
            error_detail = 'video qua 60 phut'
            return check, error_detail
        if result['is_live'] or result['was_live']:
            check = 0
            error_detail = 'video livestream'
            return check, error_detail
    return check, error_detail


def save_video_url(video_url, save_root, idx):
    ydl_opts = {
            'outtmpl': '{}/{}-%(resolution)s.%(ext)s'.format(save_root, idx)
        }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(video_url, download=True)
    return result


def get_filename_from_title(title: str, maxlen: int=10) -> str:
    filename = unidecode.unidecode(title)
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    filename = filename.translate(translator)
    filename = "_".join(filename.split()[: maxlen]).lower()
    return filename


def download_video(url, data_type):  
    root = './data_video/'
    latency = dict()
    if data_type == 7 or data_type == 1:
        print("Start download video")
        check, error_detail = check_video(url)
        if not check:
            latency['error'] = error_detail
            return None, latency
        rs = save_video_url(url, root, 'part')
        title = rs['title']
        new_title = get_filename_from_title(title)
        resolution = rs['resolution']
        ext = rs['ext']

        print("Rename video")
        os.rename(root + '{}-{}.{}'.format('part', resolution, ext), root + '{}.{}'.format(new_title, ext))
        print(new_title, ext)

        path = root + '{}.{}'.format(new_title, ext)
        new_title = f"{new_title}.{ext}"

    elif data_type == 20:
        response = requests.get(url)
        new_title = url.split('/')[-1]

        with open(root + new_title, 'wb') as f:
            f.write(response.content)
    return root + new_title