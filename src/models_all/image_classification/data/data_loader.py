import torch
from torch.utils.data import RandomSampler, BatchSampler

from data.image_dataset import LabeledDataset, UnLabeledDataset
from data.augmenter import get_strong_augment, get_weak_augment


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, unlabeled_folder, unlabeled_csv_file, train_csv_file, val_csv_file):
        class_names = ['cobac', 'tainan', 'thientai', 'chientranh', 'normal']

        train_dataset = LabeledDataset(
            annotated_path=train_csv_file,
            class_names=class_names,
            weak_augment=get_weak_augment,
            target_size=(224, 224),
            is_train=True,
            csv_separator="|",
        )

        val_dataset = LabeledDataset(
            annotated_path=val_csv_file,
            class_names=class_names,
            weak_augment=get_weak_augment,
            target_size=(224, 224),
            is_train=False,
            csv_separator="|",
        )

        unlabeled_dataset = UnLabeledDataset(
            annotated_path=unlabeled_csv_file,
            class_names=class_names,
            weak_augment=get_weak_augment,
            strong_augment=get_strong_augment,
            target_size=(224, 224),
            csv_separator="|",
        )

        num_samples = len(train_dataset)
        sampler_labeled = RandomSampler(train_dataset, replacement=True, num_samples=num_samples)
        sampler_unlabeled = RandomSampler(unlabeled_dataset, replacement=True, num_samples=3 * num_samples)

        batch_sampler_labeled = BatchSampler(sampler_labeled, batch_size=8, drop_last=False)
        batch_sampler_unlabeled = BatchSampler(sampler_unlabeled, batch_size=3 * 8, drop_last=False)

        # Set the loaders
        self.train_labeled = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler_labeled, num_workers=4)
        self.train_unlabeled = torch.utils.data.DataLoader(unlabeled_dataset, batch_sampler=batch_sampler_unlabeled, num_workers=4)
        self.test = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)