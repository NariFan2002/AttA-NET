import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
import json
import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
import transforms
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger, adjust_learning_rate, save_checkpoint
from do_train_test_val.train import train_epoch
from do_train_test_val.validation import val_epoch
from do_train_test_val.test import test_epoch
import time
import logging

opt = parse_opts()
video_transform = transforms.Compose([
                transforms.ToTensor(opt.video_norm_value)])
training_data = get_training_set(opt, spatial_transform=video_transform)
validation_data = get_validation_set(opt, spatial_transform=video_transform)
test_data = get_test_set(opt, spatial_transform=video_transform)
# val_loader = torch.utils.data.DataLoader(
#                 validation_data,
#                 batch_size=opt.batch_size,
#                 shuffle=False,
#                 num_workers=opt.n_threads,
#                 pin_memory=True)
print(len(training_data))
print(len(validation_data))
print(len(test_data))