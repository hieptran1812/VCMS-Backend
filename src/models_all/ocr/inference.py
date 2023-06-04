import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

import os
import cv2
import time
import json
import numpy as np
from PIL import Image
import torch

from tqdm import tqdm

from ocr_utils.merge_box import block_infer, merge, merge_word2line
from ocr_utils.box_utils import stitch_boxes_into_lines
from det_module.infer import sort_by_line


class Inference(object):
    def __init__(self, detect_textword, detect_textline, recognition_line, block_model):
        self.detect_textword = detect_textword
        self.detect_textline = detect_textline
        self.recognition_line = recognition_line
        self.block_model = block_model

    
    def predict(self, frames:list):
        start_time_all = time.time()
        
        block_rs = block_infer(frames, self.block_model)
        block_rs = stitch_boxes_into_lines(block_rs)
        
        det_results = self.detect_textword.detect(frames)
        line_det_results = self.detect_textline.detect(frames)
        
        det_result_list = []
        line_det_result_list = []
        
        for det_result, line_det_result in zip(det_results, line_det_results):
            all_img_crop = det_result['boundary_result']
            line_box = line_det_result['box_coordinate']
 
            if len(all_img_crop) == 0:
                result = dict()
                result['final_result'] = ''
                return result

            recog_results = self.recognition_line.single_inference(self.recognition_line.recog_model, all_img_crop, batch_mode=True, batch_size=64)

            text = []
            for idx, recog_result in enumerate(recog_results):
                word = recog_result['text']
                word = word.replace("<UKN>", "")
                word_score = recog_result['score']
                if isinstance(word_score, (list, tuple)):
                    word_score = sum(word_score) / max(1, len(word_score))
                text.append(word)

            text_results, box_text_results = merge_word2line(line_box, det_result['box_coordinate'], text)

            det_result_list.append(box_text_results)
            line_det_result_list.append(text_results)
            
        results = dict()
        
        new_rs_list, block_rs_new_list = merge(block_rs, det_result_list, line_det_result_list)
        
        for idx, (new_rs, block_rs_new, frame) in enumerate(zip(new_rs_list, block_rs_new_list, frames)):
            result = dict()
            if len(block_rs_new) != 0:
                block_convert = sorted(block_rs_new, key=lambda x: x[1])
                block_convert = [np.array(box).reshape(-1, 2).tolist() for box in block_convert]

                box_same_line = sort_by_line(block_convert, frame)
                final_result = []
                for same_line in box_same_line:
                    same_line = [np.array(box).reshape(-1).tolist() for box in same_line]
                    for box in same_line:
                        ids = block_rs_new.index(box)
                        final_result.append(new_rs[ids])

                final_block = []
                for box in block_rs_new:
                    if len(box) == 4:
                        box = [box[0], box[1], box[2], box[1], box[2], box[3], box[0], box[3]]
                        final_block.append(box)
                    else:
                        final_block.append(box)

                result['blocks'] = []
                for text_block, box_block in zip(final_result, final_block):
                    info_block = dict()
                    info_block['text'] = text_block
                    info_block['box'] = box_block
                    result['blocks'].append(info_block)
                    
            else:
                result['blocks'] = []
                final_result = '||'.join(text)

            result['final_result'] = '||'.join(final_result)
            
            results["frame_" + str(idx + 1)] = result
        
        print("Time Inference per batch frames:", time.time() - start_time_all)

        return results
    
    
    def inference_video(self, video_path, number_frame_per_batch=3, second_per_frame=5):
        torch.backends.cudnn.benchmark = True
        result = dict()
        
        video = cv2.VideoCapture(video_path)
        fps = int(video.get(cv2.CAP_PROP_FPS))
        
        print(f"Video FPS: {fps} fps")
        for index, (list_frames, list_ori_frames, time_of_frame) in enumerate(self._generate_batch_frame_from_video(video_path, 
                                                                                                         number_frame_per_batch, 
                                                                                                         second_per_frame)):
            preds = self.predict(list_frames)
            time_of_frame = "{:02d}:{:02d}".format(time_of_frame // 60, time_of_frame % 60)
            result[time_of_frame] = preds
            
        return result

    def _generate_batch_frame_from_video(self, video_path, number_frame_per_batch, second_per_frame):
        assert os.path.exists(video_path), "This video path is not exists"
        video = cv2.VideoCapture(video_path)
        fps_not_round = video.get(cv2.CAP_PROP_FPS)
        fps = int(video.get(cv2.CAP_PROP_FPS))
 
        list_frames = []
        list_ori_frames = []

        count_frame = 0
        ori_count_frame = 0

        while True:
            ret, frame = video.read()
            if ret:   
                count_frame += 1
                ori_count_frame += 1
                if count_frame % (fps * second_per_frame) == 0 and count_frame <= number_frame_per_batch * (fps * second_per_frame):
                    list_ori_frames.append(frame)
                    frame = frame[..., ::-1].copy()
                    frame = torch.tensor(frame)
                    list_frames.append(frame)

                elif count_frame > number_frame_per_batch * (fps * second_per_frame):
                    count_frame = 0
                    time_of_frame = int(float(ori_count_frame) / fps_not_round)
                    yield list_frames, np.array(list_ori_frames), time_of_frame
                    list_frames = []
                    list_ori_frames = []

            else:
                break

        video.release()