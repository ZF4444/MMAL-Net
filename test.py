#coding=utf-8
import torch
import torch.nn as nn
import sys
from tqdm import tqdm
from config import input_size, root, proposalN, channels
from utils.read_dataset import read_dataset
from utils.auto_laod_resume import auto_load_resume
from networks.model import MainNet

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

# dataset
set = 'CUB'
if set == 'CUB':
    root = './datasets/CUB_200_2011'  # dataset path
    # model path
    pth_path = "./models/cub_epoch144.pth"
    num_classes = 200
elif set == 'Aircraft':
    root = './datasets/FGVC-aircraft'  # dataset path
    # model path
    pth_path = "./models/air_epoch146.pth"
    num_classes = 100

batch_size = 10

#load dataset
_, testloader = read_dataset(input_size, batch_size, root, set)

# 定义模型
model = MainNet(proposalN=proposalN, num_classes=num_classes, channels=channels)

model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()

#加载checkpoint
if os.path.exists(pth_path):
    epoch = auto_load_resume(model, pth_path, status='test')
else:
    sys.exit('There is not a pth exist.')

print('Testing')
raw_correct = 0
object_correct = 0
model.eval()
with torch.no_grad():
    for i, data in enumerate(tqdm(testloader)):
        if set == 'CUB':
            x, y, boxes, _ = data
        else:
            x, y = data
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        local_logits, local_imgs = model(x, epoch, i, 'test', DEVICE)[-2:]
        # local
        pred = local_logits.max(1, keepdim=True)[1]
        object_correct += pred.eq(y.view_as(pred)).sum().item()

    print('\nObject branch accuracy: {}/{} ({:.2f}%)\n'.format(
            object_correct, len(testloader.dataset), 100. * object_correct / len(testloader.dataset)))