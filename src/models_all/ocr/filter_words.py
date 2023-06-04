import requests
import json
import time
from ocr_all import Ocr


class Keywords_detection(object):
    def __init__(self, ocr, keywords_config_path):
        self.keywords_config_path = keywords_config_path
        self.ocr = ocr

    def predict(self, vid_file):   
        st = time.time()
        
        res = dict()
        
        keywords_config_file = open(self.keywords_config_path)
        keywords_config = json.load(keywords_config_file)
        
        categories = keywords_config["categories"]

        prediction = self.ocr.ocr_vid(vid_file)

        final_results = set()
        text_prediction = prediction["data"]["result"]

        res['info'] = {}

        for time_frame in text_prediction.values():
            time_frame_category = []
            time_frame_text = []
            time_frame_keyword = []
            for frame in time_frame.keys():
                for category in categories.keys():
                    keywords = categories[category].split(',')
                    for keyword in keywords:
#                             print(time_frame[frame]["final_result"].lower())
                        if keyword in time_frame[frame]["final_result"].lower():
                            final_results.add(category)
                            time_frame_category.append(category)
                            time_frame_text.append(time_frame[frame]["final_result"])
                            time_frame_keyword.append(keyword)

#                 if len(time_frame_category) > 0:
#                     res['debug'][time_frame] = {}
#                     res['debug'][time_frame]["category"] = time_frame_category
#                     res['debug'][time_frame]["text"] = time_frame_text
#                     res['debug'][time_frame]["keyword"] = time_frame_keyword

        res['final_result'] = list(final_results)
        keywords_config_file.close()
        
        print(res)
        
        et = time.time()
        
        res['time_inference'] = et - st
        
        return res