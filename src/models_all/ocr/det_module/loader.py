import os
import glob
import cv2
import pyclipper
import numpy as np
import imgaug.augmenters as iaa
from shapely.geometry import Polygon

from torch.utils.data import Dataset, DataLoader
from utils import transform, crop, resize, draw_thresh_map, minmax_scaler_img


class BaseDataset(Dataset):
    def __init__(self, train_dir, train_gt_dir, ignore_tags, is_training=True,
                 image_size=640, min_text_size=8, shrink_ratio=0.4, thresh_min=0.3,
                 thresh_max=0.7, augment=True, mean=[103.939, 116.779, 123.68], debug=False):
        self.train_dir = train_dir
        self.train_gt_dir = train_gt_dir
        self.ignore_tags = ignore_tags

        self.is_training = is_training
        self.image_size = image_size
        self.min_text_size = min_text_size
        self.shrink_ratio = shrink_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max
        self.augment = augment
        if self.augment:
            self.augment = self.get_default_augment()
        self.mean = mean
        self.debug = debug
        # load data
        self.image_paths, self.gt_paths = self.load_metadata(train_dir, train_gt_dir)
        self.all_anns = self.load_all_anns(self.gt_paths)
        assert len(self.image_paths) == len(self.all_anns)

    def get_default_augment(self):
        augment_seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-10, 10)),
            iaa.Resize((0.5, 3.0))
        ])
        return augment_seq

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        anns = self.all_anns[index]

        if self.debug:
            print(image_path)
            print(len(anns))

        img = cv2.imread(image_path)[:, :, ::-1]
        if self.is_training and self.augment:
            augment_seq = self.augment.to_deterministic()
            img, anns = transform(augment_seq, img, anns)
            img, anns = crop(img, anns)

        img, anns = resize(self.image_size, img, anns)

        anns = [ann for ann in anns if Polygon(ann['poly']).buffer(0).is_valid]
        gt = np.zeros((self.image_size, self.image_size),
                      dtype=np.float32)  # batch_gts
        mask = np.ones((self.image_size, self.image_size), dtype=np.float32)
        thresh_map = np.zeros((self.image_size, self.image_size),
                              dtype=np.float32)  # batch_thresh_maps
        # batch_thresh_masks
        thresh_mask = np.zeros((self.image_size, self.image_size),
                               dtype=np.float32)

        if self.debug:
            print(type(anns), len(anns))

        ignore_tags = []
        for ann in anns:
            # i.e shape = (4, 2) / (6, 2) / ...
            poly = np.array(ann['poly'])
            height = max(poly[:, 1]) - min(poly[:, 1])
            width = max(poly[:, 0]) - min(poly[:, 0])
            polygon = Polygon(poly)

            # generate gt and mask
            if polygon.area < 1 or \
                    min(height, width) < self.min_text_size or \
                    ann['text'] in self.ignore_tags:
                ignore_tags.append(True)
                cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                continue
            else:
                # 6th equation
                distance = polygon.area * \
                    (1 - np.power(self.shrink_ratio, 2)) / polygon.length
                subject = [tuple(_l) for _l in ann['poly']]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND,
                                pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)

                if len(shrinked) == 0:
                    ignore_tags.append(True)
                    cv2.fillPoly(mask,
                                 poly.astype(np.int32)[np.newaxis, :, :], 0)
                    continue
                else:
                    shrinked = np.array(shrinked[0]).reshape(-1, 2)
                    if shrinked.shape[0] > 2 and \
                            Polygon(shrinked).buffer(0).is_valid:
                        ignore_tags.append(False)
                        cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)
                    else:
                        ignore_tags.append(True)
                        cv2.fillPoly(mask,
                                     poly.astype(np.int32)[np.newaxis, :, :],
                                     0)
                        continue

            # generate thresh map and thresh mask
            draw_thresh_map(ann['poly'], thresh_map, thresh_mask, shrink_ratio=self.shrink_ratio)

        thresh_map = thresh_map * \
            (self.thresh_max - self.thresh_min) + self.thresh_min

        img = img.astype(np.float32)
        img[..., 0] -= self.mean[0]
        img[..., 1] -= self.mean[1]
        img[..., 2] -= self.mean[2]

        img = np.transpose(img, (2, 0, 1))

        data_return = {
            "image_path": image_path,
            "img": img,
            "prob_map": gt,
            "supervision_mask": mask,
            "thresh_map": thresh_map,
            "text_area_map": thresh_mask,
        }
        # for batch_size = 1
        if not self.is_training:
            data_return["anns"] = [ann['poly'] for ann in anns]
            data_return["ignore_tags"] = ignore_tags

        # return image_path, img, gt, mask, thresh_map, thresh_mask
        return data_return


class SceneTextDatasetIter(BaseDataset):
    def __init__(self, train_dir, train_gt_dir, ignore_tags, **kwargs):
        super().__init__(train_dir, train_gt_dir, ignore_tags, **kwargs)

    @staticmethod
    def load_metadata(img_dir, gt_dir):
        img_fps = sorted(glob.glob(os.path.join(img_dir, "*")))
        gt_fps = []
        for img_fp in img_fps:
            img_id = img_fp.split("/")[-1][:-4]
            gt_fn = "{}.txt".format(img_id)
            gt_fp = os.path.join(gt_dir, gt_fn)
            assert os.path.exists(img_fp)
            gt_fps.append(gt_fp)
        assert len(img_fps) == len(gt_fps)

        return img_fps, gt_fps

    @staticmethod
    def load_all_anns(gt_fps):
        """
        Reference: https://github.com/whai362/PSENet/blob/master/dataset/ctw1500_loader.py
        """
        res = []
        for gt_fp in gt_fps:
            lines = []
            with open(gt_fp, 'r') as f:
                for line in f:
                    item = {}
                    gt = line.strip().strip('\ufeff').strip('\xef\xbb\xbf')
                    gt = list(map(int, gt.split(',')))

                    poly = list(map(int, gt))
                    poly = np.asarray(poly).reshape(-1, 2).tolist()
                    item['poly'] = poly
                    item['text'] = 'text'
                    lines.append(item)
            res.append(lines)
        return res


if __name__ == '__main__':
    import yaml
    import matplotlib.pyplot as plt
    with open('config.yaml', 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    dataset_name = cfg['dataset']['name']
    ignore_tags = cfg['data'][dataset_name]['ignore_tags']
    train_dir = cfg['data'][dataset_name]['train_dir']
    train_gt_dir = cfg['data'][dataset_name]['train_gt_dir']
    test_dir = cfg['data'][dataset_name]['test_dir']
    test_gt_dir = cfg['data'][dataset_name]['test_gt_dir']
    train_dataset = SceneTextDatasetIter(train_dir, train_gt_dir, ignore_tags, is_training=False, debug=False)
    test_iter = SceneTextDatasetIter(test_dir, test_gt_dir, ignore_tags, image_size=cfg['hps']['img_size'],
                                     is_training=False, debug=False)
    train_loader = DataLoader(train_dataset, batch_size=1,
                              shuffle=True, num_workers=1)
    test_loader = DataLoader(test_iter, batch_size=1, shuffle=True, num_workers=1)
    samples = next(iter(test_loader))
    print(samples['img'].size())  # [1, 3, 640, 640]
    print(samples['prob_map'].size())  # [1, 640, 640]
    print(samples['supervision_mask'].size())  # [1, 640, 640]
    print(samples['thresh_map'].size())  # [1, 640, 640]
    print(samples['text_area_map'].size())  # [1, 640, 640]

    plt.figure()
    plt.imshow(minmax_scaler_img(samples['img'][0].numpy().transpose(1, 2, 0)))
    plt.imshow(samples['prob_map'][0], cmap='jet', alpha=0.35)
    plt.imshow(samples['thresh_map'][0], cmap='jet', alpha=0.5)
    plt.show()