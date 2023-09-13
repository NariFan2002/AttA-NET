# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN #人脸检测库
'''
每个视频截取中间的3.6s并且均匀的选择15张人脸图片
并且保存在：对应Actor_X文件夹下面的以_facecroppad.npy为结尾的文件里面
'''
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=(720, 1280), device=device) #创建模型对象

save_frames = 15
input_fps = 30
save_length = 3.6 #seconds
save_avi = True #将脸部数据保存用于可视化目的
failed_videos = []
root = '/data1/home/fanrj/Datasets/RAVDESS'
select_distributed = lambda m, n: [i*n//m + n//(2*m) for i in range(m)] #帧的选择分布
n_processed = 0

#开始处理
for sess in tqdm(sorted(os.listdir(root))):
    #筛选出我们需要的文件夹
    if sess.startswith('Actor_'):
        for filename in os.listdir(os.path.join(root, sess)):
            if filename.endswith('.mp4') and filename.startswith('02'):#该数据集的编码规则是01代表既有音频又有视频、02是只有视频、03是只有音频
                #读取Actor-X文件夹内的文件
                # 1.计算一个视频中包含多少帧
                cap = cv2.VideoCapture(os.path.join(root, sess, filename))  #拿到文件夹视频中的内容
                #以帧为单位计算长度
                frame_number = 0
                while cap.isOpened():
                    ret,frame = cap.read() #按帧读取视频,ret,frame是获cap.read()方法的两个返回值。其中ret是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。frame就是每一帧的图像，是个三维矩阵。
                    # frame.shape：(720, 1280, 3)
                    if not ret:
                        break
                    frame_number+=1

                # 保存固定的帧数，视频长度save_length，帧数是input_fps
                cap = cv2.VideoCapture(os.path.join(root, sess, filename))  # 拿到文件夹视频中的内容
                if save_length * input_fps > frame_number:
                    skip_begin = int((frame_number - (save_length * input_fps)) // 2)
                    for i in range(skip_begin):
                        _, im = cap.read()
                frame_number = int(save_length * input_fps)
                # 确定帧的选择分布
                frames_to_select = select_distributed(save_frames, frame_number)
                save_fps = save_frames // (frame_number // input_fps)
                if save_avi:
                    out = cv2.VideoWriter(os.path.join(root, sess, filename[:-4] + '_facecroppad.avi'),
                                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), save_fps, (224, 224))

                numpy_video = []
                success = 0
                frame_ctr = 0
                while True:
                    ret, im = cap.read()
                    if not ret:
                        break
                    if frame_ctr not in frames_to_select:
                        frame_ctr += 1
                        continue
                    else:
                        frames_to_select.remove(frame_ctr)
                        frame_ctr += 1

                    try:
                        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    except:
                        failed_videos.append((sess, i))
                        break

                    temp = im[:, :, -1]
                    im_rgb = im.copy()
                    im_rgb[:, :, -1] = im_rgb[:, :, 0]
                    im_rgb[:, :, 0] = temp
                    im_rgb = torch.tensor(im_rgb)
                    im_rgb = im_rgb.to(device)

                    bbox = mtcnn.detect(im_rgb) #检测人脸的微针
                    if bbox[0] is not None:
                        bbox = bbox[0][0]
                        bbox = [round(x) for x in bbox]
                        x1, y1, x2, y2 = bbox
                    im = im[y1:y2, x1:x2, :] #将人脸截取出来
                    im = cv2.resize(im, (224, 224))
                    if save_avi:
                        out.write(im)
                    numpy_video.append(im)

                if len(frames_to_select) > 0:
                    for i in range(len(frames_to_select)):
                        if save_avi:
                            out.write(np.zeros((224, 224, 3), dtype=np.uint8))
                        numpy_video.append(np.zeros((224, 224, 3), dtype=np.uint8))
                if save_avi:
                    out.release()

                #曾经没有提取过的数据再提取一遍
                if not os.path.exists(os.path.join(root, sess, filename[:-4] + '_facecroppad.npy')):
                    np.save(os.path.join(root, sess, filename[:-4] + '_facecroppad.npy'), np.array(numpy_video))
                    #打印已经保存的数据
                    print(os.path.join(root, sess, filename[:-4] + '_facecroppad.npy'))
                if len(numpy_video) != 15:
                    print('Error', sess, filename)

        n_processed += 1
        with open('processed.txt', 'a') as f:
            f.write(sess + '\n')
        print('failed_videos:',failed_videos)

