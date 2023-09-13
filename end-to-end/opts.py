# -*- coding: utf-8 -*-
'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''

import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', default='ravdess_preprocessing/annotations.txt', type=str,
                        help='Annotation file path')
    parser.add_argument('--result_path', default='./check_point/ia_1head_moddrop_2', type=str, help='Result directory path')
    parser.add_argument('--store_name', default='ia_1head_moddrop_2', type=str, help='Name to s tore checkpoints')
    parser.add_argument('--dataset', default='RAVDESS', type=str, help='Used dataset. Currently supporting Ravdess')
    parser.add_argument('--n_classes', default=8, type=int, help='Number of classes')

    parser.add_argument('--model', default='multimodalcnn', type=str, help='')
    parser.add_argument('--num_heads', default=1, type=int, help='number of heads, in the paper 1 or 4')

    parser.add_argument('--device', default='cuda', type=str,
                        help='Specify the device to run. Defaults to cuda, fallsback to cpu')

    parser.add_argument('--sample_size', default=224, type=int, help='Video dimensions: ravdess = 224 ')
    parser.add_argument('--sample_duration', default=15, type=int, help='Temporal duration of inputs, ravdess = 15')

    parser.add_argument('--learning_rate', default=0.04, type=float,
                        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--lr_steps', default=[40, 55, 65, 70, 100, 200, 250], type=float, nargs="+", metavar='LRSteps',
                        help='epochs to decay learning rate by 10')
    parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight Decay')
    parser.add_argument('--lr_patience', default=10, type=int,
                        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch Size') #原作是8
    parser.add_argument('--n_epochs', default=100, type=int, help='Number of total epochs to run')

    parser.add_argument('--begin_epoch', default=1, type=int,
                        help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
    parser.add_argument('--resume_path', default='', type=str, help='Save data (.pth) of previous training')
    # 这个地方是用来存放 visual module weight initialization
    parser.add_argument('--pretrain_path', default='./pretrain_EfficientFace_Trained_on_AffectNet7/EfficientFace_Trained_on_AffectNet7.pth.tar', type=str,
                        help='Pretrained model (.pth), efficientface')
    parser.add_argument('--no_train', action='store_true', help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument('--no_val', action='store_true', help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument('--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=True)
    parser.add_argument('--test_subset', default='test', type=str, help='Used subset in test (val | test)')

    parser.add_argument('--n_threads', default=16, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--video_norm_value', default=255, type=int,
                        help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')

    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--fusion', default='lt', type=str, help='fusion type: lt | it | ia ｜ lt_wo_attention_block')
    parser.add_argument('--mask', type=str, help='dropout type : softhard | noise | nodropout', default='nodropout')
    parser.add_argument('--test_modality',default='both',type=str,help='val and test type: both, audio, video')
    parser.add_argument('--test_dist',default=None,type=str,help='noise,zeros,addnoise')
    parser.add_argument('--optimizer', default='SGD', type=str, help='Adam or SGD')
    parser.add_argument('--amsgrad',default=0,type=int,help='Adam amsgrad')
    args = parser.parse_args()
    #  设定数据集中样本的个数
    args.train_samples = 1680
    args.val_samples = 360
    args.test_samples = 840
    # 设定计算中特征的维度
    args.post_fusion_dim = 128*2
    args.post_audio_dim = 128
    args.post_video_dim = 128


    return args