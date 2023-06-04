import time
import torch
import json
import os
import cv2
import numpy as np
from PIL import Image


from det_module.infer import DetectDB
from inference import Inference
from mmocr.utils.ocr import MMOCR


class Ocr():
    def __init__(self, 
                 block_model='../checkpoints/detect_block.pt', 
                 det_model='../checkpoints/detect_textword.pth', 
                 line_det_model='../checkpoints/detect_textline.pth', 
                 reg_model='../checkpoints/recog_abinet.pth',
                 upload_folder='../upload'):
        
        self.detect_textword = DetectDB(weights=det_model)
        self.detect_textline = DetectDB(weights=line_det_model)
        self.recognition_line = MMOCR(det=None, 
                         recog='ABINet', 
                         recog_config='./models_all/ocr/mmocr_config/textrecog/abinet/abinet_academic.py', 
                         recog_ckpt='../checkpoints/abinet_hw.pth',
                         )
        self.block_model = torch.hub.load('ultralytics/yolov5', 'custom', path='../checkpoints/detect_block.pt')
        self.inference_all = Inference(self.detect_textword, self.detect_textline, self.recognition_line, self.block_model) 
        self.upload_folder = upload_folder
         

    def ocr_vid(self, input_vid_path):
        start = time.time()

        final_result = self.inference_all.inference_video(input_vid_path)
        infer_time = time.time() - start

        return self.create_query_result(result=final_result, infer_time=infer_time,
                                               type_data='vid', code=200, message='Success')


    def ocr_img(self, img_file=None):
        input_img = Image.open(img_file)
                
        start = time.time()
        img_arr = np.array(input_img)
        input_img = [torch.tensor(cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR))]
        final_result = self.inference_all.predict(input_img)
        infer_time = time.time() - start
        return self.create_query_result(result=final_result, infer_time=infer_time,
                                               type_data='img', code=200, message='Success')

    
    def create_query_result(self, result=None, infer_time=0., type_data='img', code=200, message='Success'):
        query_result = {
            "data": {
                "type_data": type_data,
                "result": result,
                "infer_time": infer_time,
            },
            "status": code,
            "message": message
        }
        return query_result