import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_strong_augment(target_size):
    return A.Compose([
        A.Resize(target_size[0], target_size[1]),
        A.OneOf([
            A.RandomBrightness(p=1),
            A.RandomContrast(p=1),
            A.ToGray(p=0.2),
            A.Equalize(p=0.3),
            A.Flip(p=0.5),
            A.Rotate(p=1),
            A.Posterize(p=1),
            A.Solarize(p=1),
            A.CoarseDropout(max_holes=6, min_holes=2, max_height=5, max_width=5, p=1)
        ], p=1),
        A.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
        ),
        ToTensorV2()
    ])


def get_weak_augment(target_size):
    return A.Compose([
        A.Resize(target_size[0], target_size[1]),
        A.Flip(p=0.5),
        A.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
        ),
        ToTensorV2()
    ])