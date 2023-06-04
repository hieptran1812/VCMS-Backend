import os
import cv2
import torch
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2


class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, annotated_path, class_names, weak_augment, target_size, is_train, csv_separator):
        super().__init__()
        
        assert os.path.exists(annotated_path)
        
        self.annotated_file = pd.read_csv(annotated_path, sep=csv_separator)
        self.class_names = class_names 
        
        if is_train:
            self.augment = weak_augment(target_size)
        else:
            self.augment = A.Compose([
                A.Resize(target_size[0], target_size[1]),
                A.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                ),
                ToTensorV2()
            ])
            
    def __len__(self):
        return len(self.annotated_file)
    
    def __getitem__(self, index):
        sample = self.annotated_file.iloc[index]
        image = cv2.imread(sample["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.augment(image=image)["image"]
        
        label = torch.tensor(self.class_names.index(sample["label"]))
        
        return image, label
    
    
class UnLabeledDataset(LabeledDataset):
    def __init__(self, annotated_path, class_names, weak_augment, strong_augment, target_size, csv_separator):
        super().__init__(
            annotated_path=annotated_path,
            class_names=class_names,
            weak_augment=weak_augment,
            target_size=target_size,
            is_train=True,
            csv_separator=csv_separator
        )
        
        self.strong_augment = strong_augment(target_size)
        
    def __getitem__(self, index):
        sample = self.annotated_file.iloc[index]
        image = cv2.imread(sample["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        weak_image = self.augment(image=image)["image"]
        strong_image = self.strong_augment(image=image)["image"]
        
        return weak_image, strong_image, torch.tensor([-1])  