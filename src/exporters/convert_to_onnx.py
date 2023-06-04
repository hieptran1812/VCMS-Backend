import sys
sys.path.insert(1, '/data/hieptq/brand-safety-for-video-content/src')

import os
import numpy as np
import torch
import onnxruntime as ort
from torch import nn
import onnx

from model import TinyConvNext, efficientNet, InferenceModel
from infer.classification import ContentTagging


class ONNX_operator(object):
    def __init__(self, model, onnx_model_path, checkpoint_path = None, input_shape=(1, 3, 256, 256), opset_version=12):
        if torch.cuda.is_available():
            self.device = 'cuda:1'
        else:
            self.device = 'cpu'
        self.model = model.to(self.device)
        self.onnx_model_path = onnx_model_path
        self.checkpoint_path = checkpoint_path
        self.input_shape = input_shape
        self.opset_version = opset_version
        self.input_names = ['input']
        self.output_names = ['output']
        
        if os.path.exists(self.onnx_model_path):
            self.session = ort.InferenceSession(self.onnx_model_path, providers=[('CUDAExecutionProvider', {
        'device_id': 1}), 'CPUExecutionProvider'])


    def convert_to_onnx(self):
        rand_input = torch.randn(self.input_shape).to(self.device)
        dynamic_axes_input = {}
        dynamic_axes_input[self.input_names[0]] = {0: "batch_size"}
        dynamic_axes_output = {}
        dynamic_axes_output[self.output_names[0]] = {0: "batch_size"}
        
        if (self.onnx_model_path.split('/')[-1] == 'tinyConvNext.onnx'):
            self.model = InferenceModel(self.model)
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device)["state_dict"])
        elif (self.onnx_model_path.split('/')[-1] == 'efficientnet.onnx'):
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        
        self.model.eval()
        torch.onnx.export(self.model, 
                          rand_input, 
                          self.onnx_model_path, 
                          opset_version = self.opset_version, 
                          input_names=self.input_names, 
                          output_names=self.output_names,
                          verbose=True,
                          dynamic_axes={**dynamic_axes_input, **dynamic_axes_output},
                         )
    
    def warmup_onnx(self):
        input_data = np.array(torch.randn(self.input_shape))
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        outputs = self.session.run([output_name], {input_name: input_data})
        
    def infer_from_onnx(self, input_data):
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        outputs = self.session.run([output_name], {input_name: input_data})
        ids = np.argmax(outputs[0], axis=1)
        prob = np.amax(outputs[0], axis=1)
        if self.onnx_model_path.split('/')[-1] == 'efficientnet.onnx':
            prob = torch.from_numpy(outputs[0])
            prob = torch.softmax(prob, dim=1)
            prob = prob.numpy()
            prob = np.amax(prob, axis=1)
        return ids, prob


if __name__ == '__main__':
    # Export the model to ONNX format
    tinyConvNext_onnx = ONNX_operator(TinyConvNext(), '../../models/tinyConvNext.onnx', '../../models/best.ckpt').convert_to_onnx()
    print('Model TinyConvNext has been converted to ONNX format')
    efficientnet_onnx = ONNX_operator(efficientNet(), '../../models/efficientnet.onnx', '../../models/bestval_effi.pth').convert_to_onnx()
    print('Model EfficientNet has been converted to ONNX format')
