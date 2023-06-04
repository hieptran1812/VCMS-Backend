import torch
from shapely.geometry import Polygon, MultiPoint
import numpy as np
import time


def cal_iou(bbox1, bbox2):
    bbox1 = np.array(bbox1).reshape(-1, 2)
    bbox2 = np.array(bbox2).reshape(-1, 2)
    bbox1_poly = Polygon(bbox1).convex_hull
    bbox2_poly = Polygon(bbox2).convex_hull
    union_poly = np.concatenate((bbox1, bbox2))
    if not bbox1_poly.intersects(bbox2_poly):
        iou = 0
    else:
        inter_area = bbox1_poly.intersection(bbox2_poly).area
        union_area = MultiPoint(union_poly).convex_hull.area
        if union_area == 0:
            iou = 0
        else:
            iou = float(inter_area) / bbox2_poly.area
    return iou


def merge_word2line(boxes_line, boxes_word, texts):
    match_pair_list = []
    iou_time = time.time()
    for idx, (box_word, text) in enumerate(zip(boxes_word, texts)):
        max_iou = 0
        max_match = [None, None]
        h_box = abs(box_word[-1] - box_word[1])
        for j, box_line in enumerate(boxes_line):
            devi_box = abs(box_line[1] - box_word[1])
            if devi_box > 3 * h_box:
                continue
            iou = cal_iou(box_line, box_word)
            if iou > max_iou:
                max_match[0], max_match[1] = idx, j
                max_iou = iou
        if max_match[0] is None:
            continue
        match_pair_list.append(max_match)
    
    match_pair_dict = dict()
    for match_pair in match_pair_list:
        if match_pair[1] not in match_pair_dict.keys():
            match_pair_dict[match_pair[1]] = [match_pair[0]]
        else:
            match_pair_dict[match_pair[1]].append(match_pair[0])
            
    all_word_in_line = []
    all_boxes_line = []
    for k in match_pair_dict.keys():
        idx_same_line = match_pair_dict[k]
        word_in_line = []
        box_in_line = []
        for idx in idx_same_line:
            box_in_line.append(boxes_word[idx])
        box_in_line.sort(key=lambda x:x[0])
        for box in box_in_line:
            word_in_line.append(texts[boxes_word.index(box)])
        word_in_line = ' '.join(word_in_line)
        all_word_in_line.append(word_in_line)
        all_boxes_line.append(boxes_line[k])
    
    return all_word_in_line, all_boxes_line


def get_iou(bb1, bb2):
    # bb1: yolo box
    # bb2: dbnet box
    
    new_bb2_xmin = min(bb2[0], bb2[2], bb2[4], bb2[6])
    new_bb2_xmax = max(bb2[0], bb2[2], bb2[4], bb2[6])
    new_bb2_ymin = min(bb2[1], bb2[3], bb2[5], bb2[7])
    new_bb2_ymax = max(bb2[1], bb2[3], bb2[5], bb2[7])
    new_bb2 = [new_bb2_xmin, new_bb2_ymin, new_bb2_xmax, new_bb2_ymin, new_bb2_xmax, new_bb2_ymax, new_bb2_xmin, new_bb2_ymax]
    
    x_left = max(bb1[0], new_bb2[0])
    y_top = max(bb1[1], new_bb2[1])
    x_right = min(bb1[2], new_bb2[4])
    y_bottom = min(bb1[3], new_bb2[5])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (new_bb2[4] - new_bb2[0]) * (new_bb2[5] - new_bb2[1])
    # if bb2_area == 0:
    #     bb2_area == abs(bb2[4] - bb2[0])
    iou = intersection_area / float(bb2_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def merge(yolo_rs_list, db_rs_list, text_rs_list, threshold=0.3):   
    res_all_same_box = []
    res_all_same_boxes = []

    for yolo_rs, db_rs, text_rs in zip(yolo_rs_list, db_rs_list, text_rs_list):
        all_same_box = []
        all_same_boxes = []
        correct_ids = []
        total_ids = list(range(0, len(db_rs)))
        for box_yolo in yolo_rs:
            same_box = []
            
            for idx, (box_db, text) in enumerate(zip(db_rs, text_rs)):
                if get_iou(box_yolo, box_db) > threshold:
                    same_box.append(text)
                    correct_ids.append(idx)

            same_box = ' '.join(same_box)
            if len(same_box) > 0:
                all_same_box.append(same_box)
                all_same_boxes.append(box_yolo)
        correct_ids = list(set(total_ids) - set(correct_ids))
        correct_ids.sort()
        for idx_correct in correct_ids:
            all_same_box.append(text_rs[idx_correct])
            all_same_boxes.append(db_rs[idx_correct])

        res_all_same_box.append(all_same_box)
        res_all_same_boxes.append(all_same_boxes)
    return res_all_same_box, res_all_same_boxes


def checkin(bb1, bb2):
    # bb1 < bb2
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    iou = intersection_area / float(bb1_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def block_infer(imgs:list, detect_model, thresh=0.5):
    # remove box yolo overlap
    imgs = [image.numpy() for image in imgs]
    
    all_block = detect_model(imgs).xyxy[:]

    bboxes = []
    
    for bbox in all_block:
        bbox = sorted(bbox.tolist(), key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
        list_remove = []
        for i in range(len(bbox)):
            for j in range(i + 1, len(bbox)):
                if checkin(bbox[i], bbox[j]) > thresh:
                    # overlap
                    list_remove.append(i)
                else:
                    continue
        list_remove = sorted(list(set(list_remove)))
        list_remove.reverse()
        for idx in list_remove:
            bbox.remove(bbox[idx])
        bboxes.append(bbox)
    return bboxes