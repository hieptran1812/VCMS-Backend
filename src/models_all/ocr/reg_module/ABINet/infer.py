import argparse
import logging
import os
import json
import glob
import tqdm
import torch
import PIL
import cv2
import time
import numpy as np
from natsort import natsorted
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ABINet.utils import Config, Logger, CharsetMapper

def get_model(config):
    import importlib
    names = config.model_name.split('.')
    module_name, class_name = '.'.join(names[:-1]), names[-1]
    module_name = 'ABINet.' + module_name
    cls = getattr(importlib.import_module(module_name), class_name)
    model = cls(config)
    logging.info(model)
    model = model.eval()
    return model

def preprocess(img, width, height):
    img = cv2.resize(np.array(img), (width, height))
    img = transforms.ToTensor()(img).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])
    return (img-mean[...,None,None]) / std[...,None,None]

def postprocess(output, charset, model_eval):
    def _get_output(last_output, model_eval):
        if isinstance(last_output, (tuple, list)): 
            for res in last_output:
                if res['name'] == model_eval: output = res
        else: output = last_output
        return output

    def _decode(logit):
        """ Greed decode """
        out = F.softmax(logit, dim=2)
        pt_text, pt_scores, pt_lengths = [], [], []
        for o in out:
            text = charset.get_text(o.argmax(dim=1), padding=False, trim=False)
            text = text.split(charset.null_char)[0]  # end at end-token
            pt_text.append(text)
            pt_scores.append(o.max(dim=1)[0])
            pt_lengths.append(min(len(text) + 1, charset.max_length))  # one for end-token
        return pt_text, pt_scores, pt_lengths

    output = _get_output(output, model_eval)
    logits, pt_lengths = output['logits'], output['pt_lengths']
    pt_text, pt_scores, pt_lengths_ = _decode(logits)
    
    return pt_text, pt_scores, pt_lengths_


class TextInfer(Dataset):
    def __init__(self, img_list):
        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.img_list[idx]
        # img = cv2.imread(img)
        img = cv2.resize(np.array(img), (128, 32))
        img = transforms.ToTensor()(img)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std  = torch.tensor([0.229, 0.224, 0.225])
        return (img-mean[...,None,None]) / std[...,None,None]


def load(model, file, device=None, strict=True):
    if device is None: device = 'cpu'
    elif isinstance(device, int): device = torch.device('cuda', device)
    assert os.path.isfile(file)
    state = torch.load(file, map_location=device)
    if set(state.keys()) == {'model', 'opt'}:
        state = state['model']
    model.load_state_dict(state, strict=strict)
    return model


class ABIInference:
    def __init__(self):
        config = Config('ABINet/configs/train_abinet.yaml')
        config.model_checkpoint = 'checkpoint/recog_abinet.pth'
        print("fsdfsdfsdfsdfsd", config.model_checkpoint)
        config.model_eval = 'alignment'
        config.global_phase = 'test'
        config.model_vision_checkpoint, config.model_language_checkpoint = None, None
        self.config = config
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        model = get_model(config).to(self.device)
        model = load(model, config.model_checkpoint, device=self.device)
        charset = CharsetMapper(filename=config.dataset_charset_path,
                            max_length=config.dataset_max_length + 1)
        self.model = model
        self.charset = charset

    def recognition(self, all_img):
        text_dataset = TextInfer(all_img)
        text_loader = DataLoader(text_dataset, batch_size=16, shuffle=False)
        all_result = []
        for data in text_loader:
            # print('data', data.size())
            data = data.to(self.device)
            res = self.model(data)
            pt_text, _, __ = postprocess(res, self.charset, self.config.model_eval)
            all_result += pt_text
        return all_result


"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_abinet.yaml',
                        help='path to config file')
    parser.add_argument('--input', type=str, default='figs/test')
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--checkpoint', type=str, default='best-train-abinet.pth')
    parser.add_argument('--model_eval', type=str, default='alignment', 
                        choices=['alignment', 'vision', 'language'])
    args = parser.parse_args()
    config = Config(args.config)
    if args.checkpoint is not None: config.model_checkpoint = args.checkpoint
    if args.model_eval is not None: config.model_eval = args.model_eval
    config.global_phase = 'test'
    config.model_vision_checkpoint, config.model_language_checkpoint = None, None
    
    device = 'cpu' if args.cuda < 0 else f'cuda:{args.cuda}'

    Logger.init(config.global_workdir, config.global_name, config.global_phase)
    Logger.enable_file()
    logging.info(config)

    logging.info('Construct model.')
    model = get_model(config).to(device)
     
    model = load(model, config.model_checkpoint, device=device)
    charset = CharsetMapper(filename=config.dataset_charset_path,
                            max_length=config.dataset_max_length + 1)

    if os.path.isdir(args.input):
        paths = [os.path.join(args.input, fname) for fname in os.listdir(args.input)]
    else:
        paths = glob.glob(os.path.expanduser(args.input))
        assert paths, "The input path(s) was not found"
    paths = natsorted(paths)
    
    for path in tqdm.tqdm(paths):
        img = PIL.Image.open(path).convert('RGB')
        img = preprocess(img, config.dataset_image_width, config.dataset_image_height)
        img = img.to(device)
        res = model(img)
        pt_text, _, __ = postprocess(res, charset, config.model_eval)
        logging.info(f'{path}: {pt_text[0]}')
    
    start = time.time()
    all_img = []
    for path in paths:
        img = PIL.Image.open(path).convert('RGB')
        all_img.append(img)
    text_dataset = TextInfer(all_img)
    text_loader = DataLoader(text_dataset, batch_size=8, shuffle=False)
    all_result = []
    for data in text_loader:
        # print('data', data.size())
        data = data.to(device)
        ocr_time = time.time()
        res = model(data)
        print('ocr time', time.time() - ocr_time)
        pt_text, _, __ = postprocess(res, charset, config.model_eval)
        all_result += pt_text
    
    result_dict = dict()
    for path, result in zip(paths, all_result):
        result_dict[path] = result
    with open('result.json', 'w', encoding='utf-8') as fw:
        json.dump(result_dict, fw, ensure_ascii=False)
    print('execute time', time.time() - start)


if __name__ == '__main__':
    main()
"""
