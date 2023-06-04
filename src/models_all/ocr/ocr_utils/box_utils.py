"""
merge box and ocr result in same line (word level methods)
"""
import numpy as np


def is_on_same_line(box_a, box_b, min_y_overlap_ratio=0.8):
    """Check if two boxes are on the same line by their y-axis coordinates.

    Two boxes are on the same line if they overlap vertically, and the length
    of the overlapping line segment is greater than min_y_overlap_ratio * the
    height of either of the boxes.

    Args:
        box_a (list), box_b (list): Two bounding boxes to be checked
        min_y_overlap_ratio (float): The minimum vertical overlapping ratio
                                    allowed for boxes in the same line

    Returns:
        The bool flag indicating if they are on the same line
    """
    a_y_min = np.min(box_a[1::2])
    b_y_min = np.min(box_b[1::2])
    a_y_max = np.max(box_a[1::2])
    b_y_max = np.max(box_b[1::2])

    # Make sure that box a is always the box above another
    if a_y_min > b_y_min:
        a_y_min, b_y_min = b_y_min, a_y_min
        a_y_max, b_y_max = b_y_max, a_y_max

    if b_y_min <= a_y_max:
        if min_y_overlap_ratio is not None:
            sorted_y = sorted([b_y_min, b_y_max, a_y_max])
            overlap = sorted_y[1] - sorted_y[0]
            min_a_overlap = (a_y_max - a_y_min) * min_y_overlap_ratio
            min_b_overlap = (b_y_max - b_y_min) * min_y_overlap_ratio
            return overlap >= min_a_overlap or overlap >= min_b_overlap
        else:
            return True
    return False


def stitch_boxes_into_lines(bboxes_batch:list):
    final_boxes = []
    
    for bboxes in bboxes_batch:
        all_same_line = []
        same_line = []
        bboxes = sorted(bboxes, key=lambda x: x[1])
        
        for idx in range(len(bboxes) - 1):
            same_line.append(bboxes[idx])
            if is_on_same_line(bboxes[idx], bboxes[idx + 1]) == False:
                all_same_line.append(same_line)
                same_line = []
                if idx + 1 == len(bboxes) - 1:
                    same_line.append(bboxes[idx + 1])
            else:
                if idx + 1 == len(bboxes) - 1:
                    same_line.append(bboxes[idx + 1])
            
        all_same_line.append(same_line)    
        all_same_line = [sorted(same_line, key=lambda x:x[0]) for same_line in all_same_line]
        
        boxes_processed = []
        for box_same_line in all_same_line:
            for box in box_same_line:
                boxes_processed.append(box)
                
        final_boxes.append(boxes_processed)
    
    return final_boxes