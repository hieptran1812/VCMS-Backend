import sys
sys.path.append('./models_all/ocr')
sys.path.append('./models_all/image_classification')

import time
import logging
from types import SimpleNamespace
import base64
import concurrent.futures
import json
from flask import Flask, request, Response
from flask_cors import CORS

from myutils.downloader import check_allowed_url, download_video, download_image
from myutils.general import load_config
from ocr_all import Ocr
from filter_words import Keywords_detection
from img_clf import Image_classification

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)

base_cfg = load_config('../config/default.yaml')
keywords_config_path = '../config/keywords_category.json'
cfg = SimpleNamespace(**base_cfg.__dict__)

ocr = Ocr()
img_clf = Image_classification()
filter_words = Keywords_detection(ocr, keywords_config_path)

    
api_config_path = '../config/api.json'
keywords_config_path = '../config/keywords_category.json'


@app.route('/video_content_moderation', methods=['GET'])
def video_content_moderation():
    if request.method == 'GET':
        vid_url = request.args.get('vid_url', default='', type=str)
        st = time.time()
        data_type = check_allowed_url(vid_url)
        try:
            input_vid_path = download_video(vid_url, 7)
        except Exception as ex:
            logging.exception(ex)
            return {"msg": "Download video error!"}
        
        print("======", input_vid_path)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future1 = executor.submit(img_clf.predict_video, input_vid_path)
            future2 = executor.submit(filter_words.predict, input_vid_path) 

            result = future1.result()
            result_filter_words = future2.result()

        total_time = time.time() - st
        js = {'label': list(set(result)), 'result_ocr': result_filter_words, 'total_latency': total_time}
        return json.dumps(js, ensure_ascii=False).encode('utf8')
    

@app.route('/ocr_image', methods=['GET', 'POST'])
def ocr_image():
    if request.method == 'GET':       
        img_url = request.args.get('img_url', default='', type=str)
        input_img = download_image(img_url)
        res = ocr.ocr_img(img_file=input_img)
        return res
    elif request.method == 'POST':
        file = request.files['image']
        res = ocr.ocr_img(img_file=file)
        return res
        

app.run('0.0.0.0', '8003', threaded=True)