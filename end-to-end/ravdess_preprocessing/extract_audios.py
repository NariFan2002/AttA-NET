# -*- coding: utf-8 -*-
#Librosa是一个用于音乐和音频分析的python包。它提供了创建音乐信息检索系统所需的构件。
import librosa
'''
McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto. “librosa: Audio and music signal analysis in python.” In Proceedings of the 14th python in science conference, pp. 18-25. 2015.
'''
import os
import soundfile as sf
import numpy as np

# audiofile = 'E://OpenDR_datasets//RAVDESS//Actor_19//03-01-07-02-01-02-19.wav'
'''
该文件对音频文件进行预处理，以确保它们具有相同的长度。如果length小于3.6秒，则最后用0填充。否则，它将被同等地裁剪
this file preprocess audio files to ensure they are of the same length. if length is less than 3.6 seconds, it is padded with zeros in the end. otherwise, it is equally cropped from
##both sides
'''

root = '/data1/home/fanrj/Datasets/RAVDESS/Audio_Speech_Actors_01-24'
target_time = 3.6  # sec
for actor in os.listdir(root):
    for audiofile in os.listdir(os.path.join(root, actor)):
        if not audiofile.startswith('03'):
            continue
        if not audiofile.endswith('.wav') or 'croppad' in audiofile:
            continue

        #以浮点时间序列的形式加载音频文件。音频将自动重新采样到给定的速率(默认sr=22050)。要保留文件的原生采样率，请使用sr=None。
        audios = librosa.core.load(os.path.join(root, actor, audiofile), sr=22050)

        y = audios[0] #y : np.ndarray [shape=(n,) or (2, n)] audio time series
        sr = audios[1] #sr:number > 0 [scalar] sampling rate of y
        target_length = int(sr * target_time)
        if len(y) < target_length:
            y = np.array(list(y) + [0 for i in range(target_length - len(y))])
        else:
            remain = len(y) - target_length
            y = y[remain // 2:-(remain - remain // 2)]

        sf.write(os.path.join(root, actor, audiofile[:-4] + '_croppad.wav'), y, sr)

