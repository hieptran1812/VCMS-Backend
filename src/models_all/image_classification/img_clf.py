import os
import cv2
import datetime
import numpy as np
import json
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from modeling.model import EfficientNet


class Image_classification(object):
    def __init__(self, classes=['cobac', 'tainan', 'thientai', 'chientranh', 'normal'],
                 weights_path='../checkpoints/bestval_acc0.92.pth', 
                 device="cuda:0"):
        self.classes = np.array(classes)
        self.normal_index = self.classes.tolist().index('normal')
        self.no_filter_class = []
        self.device = device
        self.model = EfficientNet.from_pretrained('efficientnet-b3', weights_path=weights_path, num_classes=5).to(self.device)
        
        self.model.eval()
        self.warm_up_model(loop=100)
        
        self.test_transform = A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            ),
            ToTensorV2(),
        ])
        

    def warm_up_model(self, loop=100):
        random_sample = torch.randn(1, 3, 224, 224).to(self.device)
        print("Start warm up model Image classification")
        for i in tqdm(range(loop)):
            with torch.no_grad():
                self.model(random_sample)   
        print("Done warm up")
        
    
    def predict_video(
        self,
        video_path,
        number_frame_per_batch=128,
        second_per_frame=1, 
        used_prob=False
    ):
        """
        Args:
            @param video_path: Path to video 
            @param number_frame_per_batch: Number frame for each sub video 
            @param second_per_frame: Time for getting one frame 
            @param used_prob: Used to prune classes which make up small percentages.
        Return 
            @return ans: List of illegal content 
        """
        
        preds = torch.tensor([])
        original_frames = None
        latency = {}
        
        for index, (batch, ori_frames) in tqdm(enumerate(self.generate_batch_frame_from_video(video_path, number_frame_per_batch, second_per_frame))):
            pred = self._predict(batch)
            original_frames = np.concatenate([original_frames, ori_frames], axis=0) if original_frames is not None else ori_frames
            preds = torch.concat([preds, pred.cpu()], axis=0)
        
        ans = self.correct_preds(preds.to(torch.int), max_consecutive_length=3)
        ans = ans.tolist()
        
        if used_prob:
            ans = self.remove_low_frequent_content(ans, lower_bound=0.15)
        
        return self.classes[ans].tolist() 
    
    
    def generate_batch_frame_from_video(self, video_path, number_frame_per_batch, second_per_frame=5):
        """
        This function is to seperate the video into frames
        Args:
            @param video_path: The path to video
            @param number_frame_per_batch: The number frame in a small part of the video 
            @param second_per_frame: Time for getting a frame 
        """
        assert os.path.exists(video_path), "This video path is not exists"
        
        video = cv2.VideoCapture(video_path)
        
        # Get video fps
        fps = int(video.get(cv2.CAP_PROP_FPS))
        print(f"Video FPS: {fps} fps")
        
        # Frame list 
        frames = []
        ori_frames = []
        
        count_frame = 0
        
        while True:
            ret, frame = video.read()
            
            if ret:
                count_frame += 1
                if count_frame % (fps * second_per_frame) == 0 and count_frame <= number_frame_per_batch * (fps * second_per_frame):
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    ori_frames.append(frame)
                    frame = self.test_transform(image=frame)["image"]
                    frames.append(frame)
                    
                elif count_frame > number_frame_per_batch * fps:
                    count_frame = 0
                
                    yield torch.stack(frames), np.array(ori_frames)
                    frames = []
                    ori_frames = []
               
            else:
#                 yield torch.stack(frames), np.array(ori_frames)
                break
                
        video.release()
        
    def create_time_video(self, seconds):
        return str(datetime.timedelta(seconds=seconds))

    
    def _calculate_abnormal_time(
        self,
        preds, 
        second_per_frame
    ):
        frame_indexes = (preds != self.normal_index).nonzero().reshape(-1,)
        frame_times = frame_indexes * second_per_frame
        
        return frame_indexes, frame_times, preds[frame_indexes.tolist()]
    
    
    def _predict(self, batch):
        """
        This function is used to predict a batch of images
        Args:
            @param batch: Tensor batch of image 
        Return:
            @return List of classes
        """
        batch = batch.to(self.device)  
        with torch.no_grad():
            preds = self.model(batch)
            
        return torch.argmax(preds, dim=-1)
    
    def correct_preds(self, preds, max_consecutive_length=3):
        """
        This function is used to filter short string of similar content in preds
        Args:
            @param preds: Original preds
            @param max_consecutive_length: Length for the string of similar content
        Return:
            @return ans: List of illegal content after filtering
        """
        
        counter = 0
        ans = preds.detach().clone()
        
        # Filter some classes which appear continuously only one or two times.
        for index, content in enumerate(ans):
            if content == self.normal_index:
                if counter > 0 and counter <= max_consecutive_length:
                    for i in range(counter):
                        ans[index - 1 - i] = self.normal_index
                
                counter = 0
                continue 
            else:
                if counter == 0:
                    counter += 1 
                else:
                    prev_content = ans[index - 1]
                    
                    if content != prev_content:
                        if prev_content not in self.no_filter_class:
                            if counter < max_consecutive_length: 
                                for i in range(counter):
                                    ans[index - 1 - i] = self.normal_index
                                counter = 1
                            else:
                                counter = 1
                    else:
                        counter += 1
                        
        return ans

    
    