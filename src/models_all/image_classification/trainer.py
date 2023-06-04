import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, accuracy_score

from helper.ema import EMA


class Trainer:
    def __init__(
        self,
        net,
        gamma,
        ema=0.999,
        device="cuda",
        class_names=['cobac', 'tainan', 'thientai', 'chientranh', 'normal']
    ):
        self.net = net 
        
        self.gamma = gamma
        # Exponential Moving Average (need read)
        self.ema = EMA(ema)
        # Creating a copy of net
        self.ema.register(self.net)
        
        self.device = device
        self.class_names = class_names
        
        self.criteron, self.optimizer, self.global_scheduler = self._init_training_setting(
            init_lr=0.0001, patience=10
        )
        
        
    def set_loader(self, labeled_loader, unlabeled_loader, test_loader):
        assert self.optimizer is not None
        
        self.labeled_loader = labeled_loader 
        self.unlabeled_loader = unlabeled_loader
        self.test_loader = test_loader
        
        self.local_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer, T_0=len(labeled_loader))
        
    def _init_training_setting(self, init_lr=0.0001, patience=10):
        """
        This function is to initialize the loss, optimizer and lr scheduler for training 
        """
        criteron = nn.CrossEntropyLoss()
#         optimizer = torch.optim.Adam(
#             self.net.parameters(),
#             lr=init_lr,
#             weight_decay=0,
#             amsgrad=False
#         )
        optimizer = torch.optim.SGD(self.net.parameters(), lr=init_lr, momentum=0.9, weight_decay=0.0005)
        global_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='max',
            factor=0.1,
            patience=patience,
            verbose=True
        )
        
        return criteron, optimizer, global_scheduler
    
    
    def save_images(self, images, labels, name):
        print("Save image")
        print(images.shape, labels.shape)
        for index, (image, label) in enumerate(zip(images, labels)):
            image = np.transpose(image, (2, 1, 0))
            cv2.imwrite("data/hieptq/test_img_clf/fix_match/labels/{}_{}_{}.jpg".format(name, index, self.class_names[label]), image)
    
    
    def train_one_epoch(self, epoch, threshold=0.9):
        assert self.labeled_loader is not None and self.unlabeled_loader is not None
        loader = zip(self.labeled_loader, self.unlabeled_loader)
        
        self.net.train()
        
        train_loss = 0
        train_tqdm = tqdm(loader, total=len(self.labeled_loader))
        
        for batch_index, (labeled_data, unlabeled_data) in enumerate(train_tqdm):
            labeled_images, y_trues = labeled_data 
            weak_images, strong_images, _ = unlabeled_data
            
            labeled_images = labeled_images.to(self.device)
            weak_images = weak_images.to(self.device)
            strong_images = strong_images.to(self.device)
            y_trues = y_trues.to(self.device)

            # Forward pass supervised 
            y_preds = self.net(labeled_images)

            # Forward pass unsupervised
            weak_y_preds = self.net(weak_images).detach()

            supervised_loss = self.criteron(y_preds, y_trues)

            with torch.no_grad():
                logits = weak_y_preds
                scores, labels = torch.max(logits, dim=-1)
                acceptable_data = scores > threshold

            if sum(acceptable_data) > 0:
                y_pseudo = labels[acceptable_data]
                strong_images = strong_images[acceptable_data]
                
#                 self.save_images(weak_images[acceptable_data].cpu().numpy(), y_pseudo.cpu().numpy(), f"{epoch}_{batch_index}")
                strong_y_preds = self.net(strong_images)

                unsupervised_loss = self.criteron(strong_y_preds, y_pseudo)
            else:
                unsupervised_loss = 0

            loss = supervised_loss + self.gamma * unsupervised_loss 
            
            assert self.local_scheduler is not None
            self.local_scheduler.step(7 * np.pi * batch_index / (16 * len(self.labeled_loader)))

            self.net.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.ema.update(self.net)
            
            train_loss += loss.item()
            train_tqdm.set_description(f'Train: Epoch [{epoch}]: Loss = {loss.item()}')
            
        return train_loss / (len(self.labeled_loader))
    
    
    def validate_one_epoch(self, epoch):
        self.net.eval()
        
        ans = []
        gt = []
        
        val_tqdm = tqdm(self.test_loader, total=len(self.test_loader))
        for batch_index, (images, labels) in enumerate(val_tqdm):
            images = images.to(self.device)
            gt += labels.tolist()
            
            preds = self.net(images)
            preds = torch.argmax(preds, dim=-1)
            ans += preds.tolist()
        
        val_acc = accuracy_score(gt, ans)
        val_f1 = f1_score(gt, ans, average="macro")
        
        print(f'Epoch {epoch}: Validation acc = {val_acc}, Validation F1 = {val_f1}')
        print(classification_report(gt, ans))
        
        return val_acc, val_f1
    
    
    def fit(self, labeled_loader, unlabeled_loader, test_loader, num_epochs, save_dir, threshold=0.9):
        best_acc = 0
        best_f1 = 0
        
        self.set_loader(labeled_loader, unlabeled_loader, test_loader)
        
        for epoch in range(num_epochs):
            train_loss = self.train_one_epoch(epoch + 1, threshold)
            print(f"Train loss: {train_loss}")
            
            val_acc, val_f1 = self.validate_one_epoch(epoch + 1)
            
            if val_acc >= best_acc:
                best_acc = val_acc
                
                print(f"Save new best acc model to {save_dir} with {val_acc} and {val_f1}")
                torch.save(self.net.state_dict(), os.path.join(save_dir, f"bestval_acc{val_acc}_f1{val_f1}.pth"))
                self.global_scheduler.step(val_acc)