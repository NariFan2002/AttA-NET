'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''
import torch
from torch.autograd import Variable
import time
from utils import AverageMeter, calculate_accuracy,BTwins, calculate_accuracy_vote
import logging
import wandb



def train_epoch_multimodal(epoch, data_loader, model, criterion, optimizer, opt,
                           epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))
    logging.debug('train at epoch {}'.format(epoch))

    model.train() #将模块设置为训练模式。

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1_vote = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1_a = AverageMeter()
    top5_a = AverageMeter()
    top1_v = AverageMeter()
    top5_v = AverageMeter()

    end_time = time.time()

    # 每个batch进行训练
    for i, (audio_inputs, visual_inputs, targets, id) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        targets = targets.to(opt.device)
        if opt.mask is not None:
            with torch.no_grad():

                if opt.mask == 'noise':
                    # 随机对音频和视频产生干扰，其中被干扰的丢失全部信息并且用该信号取代——均值为0，方差为1的正态分布(也称为标准正态分布)的随机数。
                    audio_inputs = torch.cat((audio_inputs, torch.randn(audio_inputs.size()), audio_inputs), dim=0)
                    visual_inputs = torch.cat((visual_inputs, visual_inputs, torch.randn(visual_inputs.size())), dim=0)
                    targets = torch.cat((targets, targets, targets), dim=0)
                    shuffle = torch.randperm(audio_inputs.size()[0])
                    audio_inputs = audio_inputs[shuffle]
                    visual_inputs = visual_inputs[shuffle]
                    targets = targets[shuffle]

                elif opt.mask == 'softhard':
                    coefficients = torch.randint(low=0, high=100, size=(audio_inputs.size(0), 1, 1)) / 100
                    vision_coefficients = 1 - coefficients
                    coefficients = coefficients.repeat(1, audio_inputs.size(1), audio_inputs.size(2))
                    vision_coefficients = vision_coefficients.unsqueeze(-1).unsqueeze(-1).repeat(1,
                                                                                                 visual_inputs.size(1),
                                                                                                 visual_inputs.size(2),
                                                                                                 visual_inputs.size(3),
                                                                                                 visual_inputs.size(4))

                    audio_inputs = torch.cat(
                        (audio_inputs, audio_inputs * coefficients, torch.zeros(audio_inputs.size()), audio_inputs),
                        dim=0)
                    # 这里将对视频的影响取消，训练时只保存对音频的影响
                    # visual_inputs = torch.cat((visual_inputs, visual_inputs * vision_coefficients, visual_inputs,
                    #                            torch.zeros(visual_inputs.size())), dim=0)

                    targets = torch.cat((targets, targets, targets, targets), dim=0)
                    shuffle = torch.randperm(audio_inputs.size()[0])
                    audio_inputs = audio_inputs[shuffle]
                    visual_inputs = visual_inputs[shuffle]
                    targets = targets[shuffle]

        visual_inputs = visual_inputs.permute(0, 2, 1, 3, 4)

        visual_inputs = visual_inputs.reshape(visual_inputs.shape[0] * visual_inputs.shape[1], visual_inputs.shape[2],
                                              visual_inputs.shape[3], visual_inputs.shape[4])
        #找到输入输出和标签
        audio_inputs = Variable(audio_inputs)
        visual_inputs = Variable(visual_inputs)
        targets = Variable(targets)

        '''
        outputs = model(audio_inputs, visual_inputs)
        loss = criterion(outputs, targets)   # CrossEntropyLoss
        '''
        # 更改后的多任务训练方法
        # forward
        outputs = model(audio_inputs, visual_inputs)
        outputs_av = outputs['M']
        outputs_a = outputs['A']
        outputs_v = outputs['V']
        # outputs_Feature_a=outputs['Feature_a'] #torch.Size([8, 128])
        # outputs_Feature_v = outputs['Feature_v'] #torch.Size([8, 128])
        outputs_Feature_a_for_similar = outputs['Feature_a_for_similar']
        outputs_Feature_v_for_similar = outputs['Feature_v_for_similar']
        loss1 = criterion(outputs_av, targets)   # CrossEntropyLoss
        loss2 = criterion(outputs_a, targets)
        loss3 = criterion(outputs_v, targets)
        # itcloss=BTwins(hidden_size=128,lambd=0,pj_size=128).cuda()
        # loss4 = itcloss(outputs_Feature_a_for_similar,outputs_Feature_v_for_similar)
        # print(loss4.device)
        # loss = 3*loss1+loss2+loss3+1e-6*loss4
        loss = 3*loss1+loss2+loss3
        # loss = loss3
        losses.update(loss.data, audio_inputs.size(0))
        prec1, prec5 = calculate_accuracy(outputs_av.data, targets.data, topk=(1, 5))
        prec1_vote, = calculate_accuracy_vote(torch.stack([outputs_av.data, outputs_a.data, outputs_v.data], dim=1),targets.data,
                                             topk=(1,))
        # print('3种预期值',prec1_vote,prec1,prec5)
        prec1_a, prec5_a = calculate_accuracy(outputs_a.data, targets.data, topk=(1, 5))
        prec1_v, prec5_v = calculate_accuracy(outputs_v.data, targets.data, topk=(1, 5))

        top1_vote.update(prec1_vote, audio_inputs.size(0))
        top1.update(prec1, audio_inputs.size(0))
        top5.update(prec5, audio_inputs.size(0))
        top1_a.update(prec1_a,audio_inputs.size(0))
        top5_a.update(prec5_a, audio_inputs.size(0))
        top1_v.update(prec1_v, audio_inputs.size(0))
        top5_v.update(prec5_v, audio_inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() #进行参数更新

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val.item(),
            'prec1': top1.val.item(),
            'prec5': top5.val.item(),
            'lr': optimizer.param_groups[0]['lr']
        })
        # wandb.log({
        #     'epoch': epoch,
        #     'train_loss': losses.val.item(),
        #     'train_prec1': top1.val.item(),
        #     'train_prec5': top5.val.item(),
        #     'train_a_prec1': top1_a.val.item(),
        #     'train_a_prec5': top5_a.val.item(),
        #     'train_v_prec1': top1_v.val.item(),
        #     'train_v_prec5': top5_v.val.item(),
        #     'lr': optimizer.param_groups[0]['lr']})
        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                  'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                epoch,
                i,
                len(data_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                top1=top1,
                top5=top5,
                lr=optimizer.param_groups[0]['lr']))
            logging.debug('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                  'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                epoch,
                i,
                len(data_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                top1=top1,
                top5=top5,
                lr=optimizer.param_groups[0]['lr']))


    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg.item(),
        'prec1': top1.avg.item(),
        'prec5': top5.avg.item(),
        'lr': optimizer.param_groups[0]['lr']
    })
    wandb.log({
        'epoch': epoch,
        'train_loss': losses.avg.item(),
        'train_prec1_vote_epoch_avg': top1_vote.avg.item(),
        'train_prec1_epoch_avg': top1.avg.item(),
        'train_prec5_epoch_avg': top5.avg.item(),
        'lr': optimizer.param_groups[0]['lr']
    })


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    if opt.model == 'multimodalcnn':
        train_epoch_multimodal(epoch, data_loader, model, criterion, optimizer, opt, epoch_logger, batch_logger)
        return

