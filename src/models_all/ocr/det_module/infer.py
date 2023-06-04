import sys
sys.path.insert(0, '../')

import cv2
import numpy as np
import torch

from det_module.model.db_net import DBNet
from det_module.utils import str_to_bool, read_img, test_preprocess, draw_bbox, crop_box
from det_module.post_process import SegDetectorRepresenter
from ocr_utils.box_utils import *


class DetectDB(object):
    def __init__(self, weights, thresh=0.3, box_thresh=0.5, unclip_ratio=1.5):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = DBNet().to(self.device)
        model.load_state_dict(torch.load(weights, map_location=self.device))
        self.model = model
        self.model.eval()
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio

    def detect(self, imgs):
        h_origin, w_origin = imgs[0].shape[:2]
        
        batch_img = []
        
        for img in imgs:
            img = img.numpy()
            tmp_img = test_preprocess(img, to_tensor=True, pad=False).to(self.device)
            batch_img.append(tmp_img)
        
        batch_img = torch.cat(batch_img)
        
        torch.cuda.empty_cache()
        with torch.no_grad():
            preds = self.model(batch_img)
            
        seg_obj = SegDetectorRepresenter(thresh=self.thresh, box_thresh=self.box_thresh,
                                         unclip_ratio=self.unclip_ratio)
 
        preds_list = preds.split(1, dim=0)
        
        det_results = []
        
        for img, preds in zip(imgs, preds_list):
            img = img.numpy()
            
            det_result = {}
            batch = {'shape': [(h_origin, w_origin)]}
            boxes_list, scores_list = seg_obj(batch, preds, is_output_polygon=False)

            boxes_list, scores_list = boxes_list[0].tolist(), scores_list[0]
            boxes_list.sort(key=lambda x: x[0][1])
            boxes_list_remove = []

            for boxes in boxes_list:
                if boxes[0] == boxes[2] or boxes[1] == boxes[3]:
                    continue
                else:
                    boxes_list_remove.append(boxes)

            if len(boxes_list_remove) == 0:
                det_result['img'] = img
                det_result['box_coordinate'] = []
                det_result['boundary_result'] = []
            
            else:
                sort_box_list = sort_by_line(boxes_list_remove, img)
                after_sort = []
                after_sort2 = []
                for same_row in sort_box_list:
                    for box in same_row:
                        point = np.array(box).astype(int)
                        point[0][1] = max(point[0][1] - 2, 0)
                        point[1][1] = max(point[1][1] - 2, 0)
                        point[2][1] = min(point[2][1] + 2, h_origin)
                        point[3][1] = min(point[3][1] + 2, h_origin)

                        after_sort2.append(point)
                        box = np.array(point).reshape(-1).tolist()
                        after_sort.append(box)
                        
                all_warped = []
                for index, boxes in enumerate(after_sort2):
                    warped = four_point_transform(img, boxes)
                    all_warped.append(warped)
                    
                det_result['img'] = img
                det_result['box_coordinate'] = after_sort
                det_result['boundary_result'] = all_warped
                det_results.append(det_result)
                
        return det_results


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def compute_center(box):
    top_x = (box[0][0] + box[1][0]) // 2
    top_y = (box[0][1] + box[1][1]) // 2
    bot_x = (box[2][0] + box[3][0]) // 2
    bot_y = (box[2][1] + box[3][1]) // 2
    center = [(top_x + top_y) // 2, (bot_x + bot_y) // 2]
    return center


def sort_by_line(box_info, image):
    h, w = image.shape[:2]
    scale = max(h/800, 1)
    all_same_row = []
    same_row = []
    for i in range(len(box_info)-1):
        if is_on_same_line(np.array(box_info[i+1]).reshape(-1).tolist(), np.array(box_info[i]).reshape(-1).tolist()):
            same_row.append(box_info[i])
        else:
            same_row.append(box_info[i])
            all_same_row.append(same_row)
            same_row = []
    same_row.append(box_info[-1])
    all_same_row.append(same_row)
    sort_same_row = []
    for same in all_same_row:
        same.sort(key=lambda x: x[1][0])
        sort_same_row.append(same)
    return sort_same_row