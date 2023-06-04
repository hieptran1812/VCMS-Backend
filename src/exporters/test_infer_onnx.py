import os
import torch
from convert_to_onnx import ONNX_operator
import cv2
import numpy as np
import time
from infer.classification import ContentTagging
from model import TinyConvNext, efficientNet, InferenceModel

if __name__ == '__main__':
    classes = ['cobac', 'tainan', 'thientai', 'chientranh', 'normal', 'cophandong', 'sex']

    device = None
    if torch.cuda.is_available():
        device = 'cuda:1'
    else:
        device = 'cpu'
    
    model_info_convnext_tiny = {
        "name": "convnext_tiny",
        "checkpoint_path": "../../models/best.ckpt"
    }
    model_tinyConvNext = ContentTagging(model_info_convnext_tiny, np.array(classes), device)
    print("model_tinyConvNext_onnx")
    model_tinyConvNext_onnx = ONNX_operator(TinyConvNext(), '../../models/tinyConvNext.onnx', '../../models/best.ckpt')
    print("warm")
    warmup_tinyConvNext_onnx = model_tinyConvNext_onnx.warmup_onnx()
    
    model_info_efficientNet = {
        "name": "efficientnet",
        "checkpoint_path": "../../models/bestval_effi.pth"
    }
    print("model_efficientNet_onnx")
    model_efficientNet = ContentTagging(model_info_efficientNet, np.array(classes), device)
    model_efficientNet_onnx = ONNX_operator(efficientNet(), '../../models/efficientnet.onnx', '../../models/bestval_effi.pth')
    print("warm")
    warmup_efficientNet_onnx = model_efficientNet_onnx.warmup_onnx()
    

    for img_name_file in os.listdir('./img_test'):
        if img_name_file == '.ipynb_checkpoints':
            continue
        print(img_name_file)
        
        img = cv2.imread('./img_test/' + img_name_file)
        
        # ====== Model Tiny ConvNext ======
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st = time.time()
        output_model = model_tinyConvNext.predict_image(img)
        print("Model tinyConvNext prediction:", output_model)
        et = time.time()
        print("Model tinyConvNext predict in {} seconds".format(et-st))

        st = time.time()
        img_onnx = model_tinyConvNext.test_transform(image=img)["image"]
        img_onnx = img_onnx.unsqueeze(0)
        img_onnx = np.array(img_onnx)
        ids, prob = model_tinyConvNext_onnx.infer_from_onnx(img_onnx)
        labels = np.array(classes)[ids]
        print("Model tinyConvNext ONNX prediction:", labels, prob)
        et = time.time()
        print("Model tinyConvNext ONNX predict in {} seconds\n".format(et-st))
        
        # ====== Model EfficientNet ======
        st = time.time()
        output_model = model_efficientNet.predict_image(img)
        print("Model efficientNet prediction:", output_model)
        et = time.time()
        print("Model efficientNet predict in {} seconds".format(et-st))
        
        st = time.time()
        ids, prob = model_efficientNet_onnx.infer_from_onnx(img_onnx)
        labels = np.array(classes)[ids]
        print("Model efficientNet ONNX prediction:", labels, prob)
        et = time.time()
        print("Model efficientNet ONNX predict in {} seconds\n".format(et-st))
        
        print("-----")