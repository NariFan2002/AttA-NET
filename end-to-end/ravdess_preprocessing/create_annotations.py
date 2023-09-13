# -*- coding: utf-8 -*-

import os

video_root = '/data1/home/fanrj/Datasets/RAVDESS'
audio_root = '/data1/home/fanrj/Datasets/RAVDESS/Audio_Speech_Actors_01-24'
# splits used in the paper with 5 folds
# n_folds=5
# folds =[ [[1,2,3,4],[5,6,7,8],[9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]],[[5,6,7,8],[9,10,11,12],[13,14,15,16,17,18,19,20,21,22,23,24,1,2,3,4]],[[9,10,11,12],[13,14,15,16],[17,18,9,20,21,22,23,24,1,2,3,4,5,6,7,8]],[[13,14,15,16],[17,18,19,20],[21,22,23,24,1,2,3,4,5,6,7,8,9,10,11,12]],[[17,18,19,20],[21,22,23,24],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]]

n_folds = 1
folds = [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]]]
for fold in range(n_folds):
    fold_ids = folds[fold]
    test_ids, val_ids, train_ids = fold_ids

    # annotation_file = 'annotations_croppad_fold'+str(fold+1)+'.txt'
    annotation_file = 'annotations.txt'
    # 设置变量用来计数不同数据集中的样本个数
    test_num = 0
    train_num = 0
    val_num = 0
    for i, actor in enumerate(os.listdir(video_root)):
        if actor.startswith('Actor_'):
            for video in os.listdir(os.path.join(video_root, actor)):
                if not video.endswith('.npy') or 'croppad' not in video:
                    continue
                label = str(int(video.split('-')[2]))
                audio = '03' + video.split('_face')[0][2:] + '_croppad.wav'
                if i in train_ids:
                    with open(annotation_file, 'a') as f:
                        f.write(os.path.join(video_root, actor, video) + ';' + os.path.join(audio_root, actor,
                                                                                      audio) + ';' + label +';'+ str(train_num) +';training' + '\n')
                    train_num += 1


                elif i in val_ids:
                    with open(annotation_file, 'a') as f:
                        f.write(os.path.join(video_root, actor, video) + ';' + os.path.join(audio_root, actor,
                                                                                      audio) + ';' + label +';'+ str(val_num) + ';validation' + '\n')
                    val_num += 1

                else:

                    with open(annotation_file, 'a') as f:
                        f.write(os.path.join(video_root, actor, video) + ';' + os.path.join(audio_root, actor,
                                                                                      audio) + ';' + label +';'+ str(test_num) + ';testing' + '\n')
                    test_num += 1
    print("train_num：",train_num," val_num：",val_num," test_num:",test_num)
    # 运行结果：train_num： 1680  val_num： 360  test_num: 840
