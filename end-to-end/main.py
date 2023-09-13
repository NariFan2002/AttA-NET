# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:07:29 2021

@author: chumache
"""
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
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
import wandb

if __name__ == '__main__':
    opt = parse_opts()
    n_folds = 1
    test_accuracies = []

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="multimodal-emotion-recognition-AV-RAVDESS",
        tags=["batch_size 32","SGD","compress"],
        name='lt train nodropout muti-task',
        # track hyperparameters and run metadata
        config= vars(opt) # vars将opt的Namespace object变成dict #保存网络参数信息
    )

    if opt.device != 'cpu':
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('最终选择的计算设备:',opt.device)

    pretrained = opt.pretrain_path != 'None'

    opt.result_path = 'result/'+'res_'+str(time.asctime( time.localtime(time.time()) ))
    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)
    logging.basicConfig(filename=os.path.join(opt.result_path,'运行记录' + str(time.asctime(time.localtime(time.time()))) + '.log'),
                        level=logging.DEBUG, format=LOG_FORMAT)
    logging.debug('最终选择的计算设备:' + opt.device)

    opt.arch = '{}'.format(opt.model)
    opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.sample_duration)])

    for fold in range(n_folds):
        # if opt.dataset == 'RAVDESS':
        #    opt.annotation_path = '/lustre/scratch/chumache/ravdess-develop/annotations_croppad_fold'+str(fold+1)+'.txt'

        print(opt)
        logging.debug(opt)
        with open(os.path.join(opt.result_path, 'opts' + str(time.asctime(time.localtime(time.time()))) + str(fold) + '.json'), 'a') as opt_file:
            json.dump(vars(opt), opt_file)
            #'a' 打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。也就是说，新的内容将会被写入到已有内容之后。如果该文件不存在，创建新文件进行写入。
        torch.manual_seed(opt.manual_seed)
        # 载入模型和参数
        model, parameters = generate_model(opt)

        # 确定cost函数的计算方法
        criterion = nn.CrossEntropyLoss() # loss = CrossEntropyLoss（）
        criterion = criterion.to(opt.device)

        # 配置训练用到的数据集，logger，和优化器
        if not opt.no_train:
            # 设置数据集
            video_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotate(),
                transforms.ToTensor(opt.video_norm_value)])
            training_data = get_training_set(opt, spatial_transform=video_transform)
            train_loader = torch.utils.data.DataLoader(
                training_data,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.n_threads,
                pin_memory=True)
            # 设置logger记录部分
            train_logger = Logger(
                os.path.join(opt.result_path, 'train' + str(fold) + '.log'),
                ['epoch', 'loss', 'prec1', 'prec5', 'lr'])
            train_batch_logger = Logger(
                os.path.join(opt.result_path, 'train_batch' + str(fold) + '.log'),
                ['epoch', 'batch', 'iter', 'loss', 'prec1', 'prec5', 'lr'])

            # 设置模型训练的优化器
            assert opt.optimizer in ["SGD","Adam"]
            if opt.optimizer=="SGD":
                optimizer = optim.SGD(
                    filter(lambda p: p.requires_grad, model.parameters()),  # 传入模型的参数
                    lr=opt.learning_rate,  # 初始权重
                    momentum=opt.momentum,  # 设置动量参数，梯度采用累加的算法：震荡的方向的梯度互相抵消，梯度小的方向逐渐累加
                    dampening=opt.dampening,
                    weight_decay=opt.weight_decay,  # 权重衰减
                    nesterov=False)
            elif opt.optimizer=="Adam":
                optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),  # 传入模型的参数
                    lr=opt.learning_rate,  # 初始权重
                    betas=(0.9,0.999), #动量参数和衰减系数
                    weight_decay=opt.weight_decay,  # 权重衰减
                    amsgrad=opt.amsgrad
                )
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=opt.lr_patience)


        # 配置val和test用到的验证集和logger
        if not opt.no_val:
            video_transform = transforms.Compose([
                transforms.ToTensor(opt.video_norm_value)])

            validation_data = get_validation_set(opt, spatial_transform=video_transform)

            val_loader = torch.utils.data.DataLoader(
                validation_data,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)

            val_logger = Logger(
                os.path.join(opt.result_path, 'val' + str(fold) + '.log'), ['epoch', 'loss', 'prec1', 'prec5'])
            test_logger = Logger(
                os.path.join(opt.result_path, 'test' + str(fold) + '.log'), ['epoch', 'loss', 'prec1', 'prec5'])

        best_prec1 = 0
        best_loss = 1e10
        if opt.resume_path:
            print('loading checkpoint {}'.format(opt.resume_path))
            checkpoint = torch.load(opt.resume_path)
            assert opt.arch == checkpoint['arch']
            best_prec1 = checkpoint['best_prec1']
            opt.begin_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])

        # 开始训练，每个epoch训练一次
        for i in range(opt.begin_epoch, opt.n_epochs + 1):
            # 训练
            if not opt.no_train:
                adjust_learning_rate(optimizer, i, opt)
                train_epoch(i, train_loader, model, criterion, optimizer, opt,
                            train_logger, train_batch_logger)
                state = {
                    'epoch': i,
                    'arch': opt.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1,
                    'model':model
                }
                # 不会保存bast_prec的模型参数
                save_checkpoint(state, False, opt, fold)
                logging.debug('该轮训练的数据已保存')
            # 验证
            if not opt.no_val:
                validation_loss, prec1 = val_epoch(epoch=i, data_loader=val_loader, model=model, criterion=criterion, opt=opt,
                                                   logger=val_logger,modality=opt.test_modality, dist=opt.test_dist)
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                state = {
                    'epoch': i,
                    'arch': opt.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1,
                    'model': model
                }
                scheduler.step(validation_loss) #当metric(validation_loss)停止改进时，降低学习率(lr)。
                save_checkpoint(state, is_best, opt, fold)
                logging.debug('该轮验证的数据已保存')
                wandb.log({'validation_best_prec1':best_prec1})

        # 开始验证
        if opt.test:

            test_logger = Logger(
                os.path.join(opt.result_path, 'test' + str(fold) + '.log'), ['epoch', 'loss', 'prec1', 'prec5'])

            video_transform = transforms.Compose([
                transforms.ToTensor(opt.video_norm_value)])

            test_data = get_test_set(opt, spatial_transform=video_transform)
            # test_data = get_training_set(opt, spatial_transform=video_transform)
            # test_data = get_validation_set(opt, spatial_transform=video_transform)
            # 如果是当次训练以后那么load最新的best model
            logging.debug('load best model:'+'%s/%s_best' % (opt.result_path, opt.store_name) + str(fold) + '.pth')
            if os.path.exists('%s/%s_best' % (opt.result_path, opt.store_name) + str(fold) + '.pth'):
                best_state = torch.load('%s/%s_best' % (opt.result_path, opt.store_name) + str(fold) + '.pth')
                # best_state = torch.load('%s/%s' % (opt.result_path, opt.store_name)+ '.pth')
                model.load_state_dict(best_state['state_dict'])

            test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)

            test_loss, test_prec1 = test_epoch(-1, test_loader, model, criterion, opt,
                                              test_logger,modality=opt.test_modality, dist=opt.test_dist)

            with open(os.path.join(opt.result_path, 'test_set_bestval' + str(fold) + '.txt'), 'a') as f:
                f.write('Prec1: ' + str(test_prec1) + '; Loss: ' + str(test_loss))
            logging.debug('test_fold'+str(fold)+':'+'Prec1: ' + str(test_prec1) + '; Loss: ' + str(test_loss))
            test_accuracies.append(test_prec1)




    with open(os.path.join(opt.result_path, 'test_set_bestval.txt'), 'a') as f:
        f.write(
            'Prec1: ' + str(np.mean(np.array(test_accuracies))) + '+' + str(np.std(np.array(test_accuracies))) + '\n')
    logging.debug('test_fold_average:'+'Prec1: ' + str(np.mean(np.array(test_accuracies))) + '+' + str(np.std(np.array(test_accuracies))) + '\n')

    wandb.finish()