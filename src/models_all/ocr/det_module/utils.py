import os
import gc
import glob
import time
import random
import imageio
import logging
from functools import wraps
import imgaug
import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import torch
import torchvision.utils as torch_utils

from det_module.post_process import SegDetectorRepresenter

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'


def setup_determinism(seed=42):
    """
    https://github.com/pytorch/pytorch/issues/7068#issuecomment-487907668
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_logger(logger_name='dbtext', log_file_path=None):
    logging._warn_preinit_stderr = 0
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s %(name)s %(levelname)s: %(message)s')

    if log_file_path is not None:
        file_handle = logging.FileHandler(log_file_path)
        file_handle.setFormatter(formatter)
        logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)

    return logger


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(">>> Function {}: {}'s".format(func.__name__, end - start))
        return result

    return wrapper


def to_device(batch, device='cuda'):
    new_batch = []

    for ele in batch:
        if isinstance(ele, torch.Tensor):
            new_batch.append(ele.to(device))
        else:
            new_batch.append(ele)
    return new_batch


def dict_to_device(batch, device='cuda'):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


def to_list_tuples_coords(anns):
    new_anns = []
    for ann in anns:
        points = []
        for x, y in ann:
            points.append((x[0].tolist(), y[0].tolist()))
        new_anns.append(points)
    return new_anns


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def str_to_bool(value):
    if value.lower() in {'False', 'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'True', 'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError('{} is not a valid boolean value'.format(value))


def minmax_scaler_img(img):
    img = ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype(
        'uint8')  # noqa
    return img


def visualize_tfb(tfb_writer,
                  imgs,
                  preds,
                  global_steps,
                  thresh=0.5,
                  mode="TRAIN"):
    # origin img
    # imgs.shape = (batch_size, 3, image_size, image_size)
    imgs = torch.stack([
        torch.Tensor(
            minmax_scaler_img(img_.to('cpu').numpy().transpose((1, 2, 0))))
        for img_ in imgs
    ])
    imgs = torch.Tensor(imgs.numpy().transpose((0, 3, 1, 2)))
    imgs_grid = torch_utils.make_grid(imgs)
    imgs_grid = torch.unsqueeze(imgs_grid, 0)
    # imgs_grid.shape = (3, image_size, image_size * batch_size)
    tfb_writer.add_images('{}/origin_imgs'.format(mode), imgs_grid,
                          global_steps)

    # pred_prob_map / pred_thresh_map
    pred_prob_map = preds[:, 0, :, :]
    pred_thred_map = preds[:, 1, :, :]
    pred_prob_map[pred_prob_map <= thresh] = 0
    pred_prob_map[pred_prob_map > thresh] = 1

    # make grid
    pred_prob_map = pred_prob_map.unsqueeze(1)
    pred_thred_map = pred_thred_map.unsqueeze(1)

    probs_grid = torch_utils.make_grid(pred_prob_map, padding=0)
    probs_grid = torch.unsqueeze(probs_grid, 0)
    probs_grid = probs_grid.detach().to('cpu')

    thres_grid = torch_utils.make_grid(pred_thred_map, padding=0)
    thres_grid = torch.unsqueeze(thres_grid, 0)
    thres_grid = thres_grid.detach().to('cpu')

    tfb_writer.add_images('{}/prob_imgs'.format(mode), probs_grid,
                          global_steps)
    tfb_writer.add_images('{}/thres_imgs'.format(mode), thres_grid,
                          global_steps)


def test_resize(img, size=640, pad=False):
    h, w, c = img.shape
    scale_w = size / w
    scale_h = size / h
    scale = min(scale_w, scale_h)
    h = int(h * scale)
    w = int(w * scale)

    new_img = None
    if pad:
        new_img = np.zeros((size, size, c), img.dtype)
        new_img[:h, :w] = cv2.resize(img, (w, h))
    else:
        new_img = cv2.resize(img, (w, h))

    return new_img


def read_img(img_fp):
    img = cv2.imread(img_fp)[:, :, ::-1]
    h_origin, w_origin, _ = img.shape
    return img, h_origin, w_origin


def test_preprocess(img,
                    mean=[103.939, 116.779, 123.68],
                    to_tensor=True,
                    pad=False):
    img = test_resize(img, size=1024, pad=pad)

    img = img.astype(np.float32)
    img[..., 0] -= mean[0]
    img[..., 1] -= mean[1]
    img[..., 2] -= mean[2]
    img = np.expand_dims(img, axis=0)

    if to_tensor:
        img = torch.Tensor(img.transpose(0, 3, 1, 2))

    return img


def draw_bbox(img, result, color=(255, 0, 0), thickness=3):
    """
    :input: RGB img
    """
    if isinstance(img, str):
        img = cv2.imread(img)
    img = img.copy()
    h, w = img.shape[:2]
    for point in result:
        point = point.astype(int)
        point[0][0] = max(point[0][0] - 5, 0)
        point[0][1] = max(point[0][1] - 5, 0)
        point[1][0] = max(point[1][0] + 5, 0)
        point[1][1] = min(point[1][1] - 5, h)
        point[2][0] = min(point[2][0] + 5, w)
        point[2][1] = min(point[2][1] + 5, h)
        point[3][0] = max(point[3][0] - 5, 0)
        point[3][1] = min(point[3][1] + 5, h)
        cv2.polylines(img, [point], True, color, thickness)
    return img


def crop_box(src_img, box, long_edge_pad_ratio=0.4, short_edge_pad_ratio=0.2):
    """Crop text region with their bounding box.
    Args:
        src_img (np.array): The original image.
        box (list[float | int]): Points of quadrangle.
        long_edge_pad_ratio (float): Box pad ratio for long edge
            corresponding to font size.
        short_edge_pad_ratio (float): Box pad ratio for short edge
            corresponding to font size.
    """
    assert len(box) == 8
    assert 0. <= long_edge_pad_ratio < 1.0
    assert 0. <= short_edge_pad_ratio < 1.0

    h, w = src_img.shape[:2]
    points_x = np.clip(np.array(box[0::2]), 0, w)
    points_y = np.clip(np.array(box[1::2]), 0, h)

    box_width = np.max(points_x) - np.min(points_x)
    box_height = np.max(points_y) - np.min(points_y)
    font_size = min(box_height, box_width)

    if box_height < box_width:
        horizontal_pad = long_edge_pad_ratio * font_size
        vertical_pad = short_edge_pad_ratio * font_size
    else:
        horizontal_pad = short_edge_pad_ratio * font_size
        vertical_pad = long_edge_pad_ratio * font_size

    left = np.clip(int(np.min(points_x) - horizontal_pad), 0, w)
    top = np.clip(int(np.min(points_y) - vertical_pad), 0, h)
    right = np.clip(int(np.max(points_x) + horizontal_pad), 0, w)
    bottom = np.clip(int(np.max(points_y) + vertical_pad), 0, h)

    dst_img = src_img[top:bottom, left:right]

    return dst_img


def draw_polygon_image(img, result, color=(255, 0, 0), thickness=3):
    if isinstance(img, str):
        img = cv2.imread(img)
    img = img.copy()
    for point in result:
        point = point.astype(int)
        cv2.polylines(img, [point], True, color, thickness)
    return img


def visualize_heatmap(args, img_fn, tmp_img, tmp_pred):
    pred_prob = tmp_pred[0]
    pred_prob[pred_prob <= args.prob_thred] = 0
    pred_prob[pred_prob > args.prob_thred] = 1

    np_img = minmax_scaler_img(tmp_img[0].to(device).numpy().transpose(
        (1, 2, 0)))
    plt.imshow(np_img)
    plt.imshow(pred_prob, cmap='jet', alpha=args.alpha)
    img_fn = "heatmap_result_{}".format(img_fn)
    plt.savefig(os.path.join(args.save_dir, img_fn),
                dpi=200,
                bbox_inches='tight')
    gc.collect()


def visualize_polygon(args, img_fn, origin_info, batch, preds, vis_char=False):
    img_origin, h_origin, w_origin = origin_info
    seg_obj = SegDetectorRepresenter(thresh=args.thresh,
                                     box_thresh=args.box_thresh,
                                     unclip_ratio=args.unclip_ratio)
    box_list, score_list = seg_obj(batch, preds, is_output_polygon=args.is_output_polygon)
    box_list, score_list = box_list[0], score_list[0]

    if len(box_list) > 0:
        if args.is_output_polygon:
            idx = [x.sum() > 0 for x in box_list]
            box_list = [box_list[i] for i, v in enumerate(idx) if v]
            score_list = [score_list[i] for i, v in enumerate(idx) if v]
        else:
            idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0
            box_list, score_list = box_list[idx], score_list[idx]
    else:
        box_list, score_list = [], []

    tmp_img = draw_bbox(img_origin, np.array(box_list))
    tmp_pred = cv2.resize(preds[0, 0, :, :].cpu().numpy(),
                          (w_origin, h_origin))

    # https://stackoverflow.com/questions/42262198
    h_, w_ = 32, 100
    if not args.is_output_polygon and vis_char:

        char_img_fps = glob.glob(os.path.join("./tmp/reconized", "*"))
        for char_img_fp in char_img_fps:
            os.remove(char_img_fp)

        for index, (box_list_,
                    score_list_) in enumerate(zip(box_list,
                                                  score_list)):  # noqa
            src_pts = np.array(box_list_.tolist(), dtype=np.float32)
            dst_pts = np.array([[0, 0], [w_, 0], [w_, h_], [0, h_]],
                               dtype=np.float32)
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warp = cv2.warpPerspective(img_origin, M, (w_, h_))
            imageio.imwrite("./tmp/reconized/word_{}.jpg".format(index), warp)

    plt.imshow(tmp_img)
    plt.imshow(tmp_pred, cmap='inferno', alpha=args.alpha)
    if args.is_output_polygon:
        img_fn = "poly_result_{}".format(img_fn)
    else:
        img_fn = "rect_result_{}".format(img_fn)
    plt.savefig(os.path.join(args.save_path, img_fn),
                dpi=200,
                bbox_inches='tight')
    gc.collect()


def draw_thresh_map(polygon, canvas, mask, shrink_ratio=0.4):
    polygon = np.array(polygon)
    assert polygon.ndim == 2
    assert polygon.shape[1] == 2

    polygon_shape = Polygon(polygon)
    if polygon_shape.area <= 0:
        return
    distance = polygon_shape.area * (
        1 - np.power(shrink_ratio, 2)) / polygon_shape.length
    subject = [tuple(l) for l in polygon]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    padded_polygon = np.array(padding.Execute(distance)[0])
    cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

    xmin = padded_polygon[:, 0].min()
    xmax = padded_polygon[:, 0].max()
    ymin = padded_polygon[:, 1].min()
    ymax = padded_polygon[:, 1].max()
    width = xmax - xmin + 1
    height = ymax - ymin + 1

    polygon[:, 0] = polygon[:, 0] - xmin
    polygon[:, 1] = polygon[:, 1] - ymin

    xs = np.broadcast_to(
        np.linspace(0, width - 1, num=width).reshape(1, width),
        (height, width))
    ys = np.broadcast_to(
        np.linspace(0, height - 1, num=height).reshape(height, 1),
        (height, width))

    distance_map = np.zeros((polygon.shape[0], height, width),
                            dtype=np.float32)
    for i in range(polygon.shape[0]):
        j = (i + 1) % polygon.shape[0]
        absolute_distance = compute_distance(xs, ys, polygon[i], polygon[j])
        distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
    distance_map = np.min(distance_map, axis=0)

    xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
    xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
    ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
    ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)

    # canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
    #     1 - distance_map[ymin_valid - ymin:ymax_valid - ymin + 1,  # add 1
    #                      xmin_valid - xmin:xmax_valid - xmin + 1  # add 1
    #                      ],
    #     canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

    canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
        1 - distance_map[ymin_valid - ymin:ymax_valid - ymax + height,
                         xmin_valid - xmin:xmax_valid - xmax + width],
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])


def compute_distance(xs, ys, point_1, point_2):
    square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
    square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
    square_distance = np.square(point_1[0] -
                                point_2[0]) + np.square(point_1[1] -
                                                        point_2[1])

    cosin = (square_distance - square_distance_1 - square_distance_2) / \
            (2 * np.sqrt(square_distance_1 * square_distance_2))
    square_sin = 1 - np.square(cosin)
    square_sin = np.nan_to_num(square_sin)
    result = np.sqrt(square_distance_1 * square_distance_2 * square_sin /
                     square_distance)

    result[cosin < 0] = np.sqrt(np.fmin(square_distance_1,
                                        square_distance_2))[cosin < 0]
    return result


def split_regions(axis):
    regions = []
    min_axis_index = 0
    for i in range(1, axis.shape[0]):
        if axis[i] != axis[i - 1] + 1:
            region = axis[min_axis_index:i]
            min_axis_index = i
            regions.append(region)
    return regions


def random_select(axis):
    xx = np.random.choice(axis, size=2)
    xmin = np.min(xx)
    xmax = np.max(xx)
    return xmin, xmax


def region_wise_random_select(regions):
    selected_index = list(np.random.choice(len(regions), 2))
    selected_values = []
    for index in selected_index:
        axis = regions[index]
        xx = int(np.random.choice(axis, size=1))
        selected_values.append(xx)
    xmin = min(selected_values)
    xmax = max(selected_values)
    return xmin, xmax


# utils build dataset
def crop(image, anns, max_tries=10, min_crop_side_ratio=0.1):
    h, w, _ = image.shape
    h_array = np.zeros(h, dtype=np.int32)
    w_array = np.zeros(w, dtype=np.int32)
    for ann in anns:
        points = np.round(ann['poly'], decimals=0).astype(np.int32)
        minx = np.min(points[:, 0])
        maxx = np.max(points[:, 0])
        w_array[minx:maxx] = 1
        miny = np.min(points[:, 1])
        maxy = np.max(points[:, 1])
        h_array[miny:maxy] = 1
    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]

    if len(h_axis) == 0 or len(w_axis) == 0:
        return image, anns

    h_regions = split_regions(h_axis)
    w_regions = split_regions(w_axis)

    for i in range(max_tries):
        if len(w_regions) > 1:
            xmin, xmax = region_wise_random_select(w_regions)
        else:
            xmin, xmax = random_select(w_axis)
        if len(h_regions) > 1:
            ymin, ymax = region_wise_random_select(h_regions)
        else:
            ymin, ymax = random_select(h_axis)

        if xmax - xmin < min_crop_side_ratio * w or ymax - ymin < min_crop_side_ratio * h:
            # area too small
            continue
        new_anns = []
        for ann in anns:
            poly = np.array(ann['poly'])
            if not (poly[:, 0].min() > xmax or poly[:, 0].max() < xmin
                    or poly[:, 1].min() > ymax or poly[:, 1].max() < ymin):
                poly[:, 0] -= xmin
                poly[:, 0] = np.clip(poly[:, 0], 0., (xmax - xmin - 1) * 1.)
                poly[:, 1] -= ymin
                poly[:, 1] = np.clip(poly[:, 1], 0., (ymax - ymin - 1) * 1.)
                new_ann = {'poly': poly.tolist(), 'text': ann['text']}
                new_anns.append(new_ann)

        if len(new_anns) > 0:
            return image[ymin:ymax, xmin:xmax], new_anns

    return image, anns


def resize(size, image, anns):
    h, w, c = image.shape
    scale_w = size / w
    scale_h = size / h
    scale = min(scale_w, scale_h)
    h = int(h * scale)
    w = int(w * scale)
    padimg = np.zeros((size, size, c), image.dtype)
    padimg[:h, :w] = cv2.resize(image, (w, h))
    new_anns = []
    for ann in anns:
        poly = np.array(ann['poly']).astype(np.float64)
        poly *= scale
        new_ann = {'poly': poly.tolist(), 'text': ann['text']}
        new_anns.append(new_ann)
    return padimg, new_anns


def transform(aug, image, anns):
    image_shape = image.shape
    image = aug.augment_image(image)
    new_anns = []
    for ann in anns:
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in ann['poly']]
        keypoints = aug.augment_keypoints(
            [imgaug.KeypointsOnImage(keypoints,
                                     shape=image_shape)])[0].keypoints
        poly = [(min(max(0, p.x),
                     image.shape[1] - 1), min(max(0, p.y), image.shape[0] - 1))
                for p in keypoints]
        new_ann = {'poly': poly, 'text': ann['text']}
        new_anns.append(new_ann)
    return image, new_anns
