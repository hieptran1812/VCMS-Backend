import timm
import torch
import yaml

from data.data_loader import DataLoader
from modeling.model import EfficientNet
from trainer import Trainer


if __name__ == '__main__':
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
        unlabeled_folder = config["data"]["unlabeled_folder"]
        unlabeled_csv_file = config["data"]["unlabeled_csv_file"]
        train_csv_file = config["data"]["train_csv_file"]
        val_csv_file = config["data"]["val_csv_file"]
        device = config["device"]
    
    dataloader = DataLoader(unlabeled_folder, unlabeled_csv_file, train_csv_file, val_csv_file)
    
    train_labeled = dataloader.train_labeled
    train_unlabeled = dataloader.train_unlabeled
    test = dataloader.test

    model = EfficientNet.from_pretrained('efficientnet-b3',  num_classes=5)

    trainer = Trainer(
        net=model.to(device),
        gamma=1,
        ema=0.999,
        device=device
    )
    
    trainer.fit(
        labeled_loader=train_labeled, 
        unlabeled_loader=train_unlabeled, 
        test_loader=test, 
        num_epochs=1000, 
        save_dir="/data/hieptq/test_img_clf/weights", 
        threshold=0.95
    )