# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from tqdm import tqdm
import torch

"""
这个代码主要是为了检测人脸图片的正确性
"""
faces_15 = os.read('/data1/home/fanrj/Datasets/RAVDESS/Actor_01/02-01-08-01-02-02-01_facecroppad.npy')
