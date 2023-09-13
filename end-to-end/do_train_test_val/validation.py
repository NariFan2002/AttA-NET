'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''
import torch
from torch.autograd import Variable
import time
from utils import AverageMeter, calculate_accuracy,BTwins, calculate_accuracy_vote
import logging
import wandb

def val_epoch_multimodal(epoch, data_loader, model, criterion, opt, logger, modality='both', dist=None):
    # for evaluation with single modality, specify which modality to keep and which distortion to apply for the other modaltiy:
    # 'noise', 'addnoise' or 'zeros'. for paper procedure, with 'softhard' mask use 'zeros' for evaluation, with 'noise' use 'noise'
    print('validation at epoch {}'.format(epoch))
    logging.debug('validation at epoch {}'.format(epoch) + 'or test when epoch == 10000')
    assert modality in ['both', 'audio', 'video']
    model.eval()

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
    for i, (inputs_audio, inputs_visual, targets, id) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if modality == 'audio':
            print('Skipping video modality')
            if dist == 'noise':
                print('Evaluating with full noise')
                inputs_visual = torch.randn(inputs_visual.size())
            elif dist == 'addnoise':  # opt.mask == -4:
                print('Evaluating with noise')
                inputs_visual = inputs_visual + (
                            torch.mean(inputs_visual) + torch.std(inputs_visual) * torch.randn(inputs_visual.size()))
            elif dist == 'zeros':
                inputs_visual = torch.zeros(inputs_visual.size())
            else:
                print('UNKNOWN DIST!')
        elif modality == 'video':
            print('Skipping audio modality')
            if dist == 'noise':
                print('Evaluating with noise')
                inputs_audio = torch.randn(inputs_audio.size())
            elif dist == 'addnoise':  # opt.mask == -4:
                print('Evaluating with added noise')
                inputs_audio = inputs_audio + (
                            torch.mean(inputs_audio) + torch.std(inputs_audio) * torch.randn(inputs_audio.size()))

            elif dist == 'zeros':
                inputs_audio = torch.zeros(inputs_audio.size())
        inputs_visual = inputs_visual.permute(0, 2, 1, 3, 4)
        inputs_visual = inputs_visual.reshape(inputs_visual.shape[0] * inputs_visual.shape[1], inputs_visual.shape[2],
                                              inputs_visual.shape[3], inputs_visual.shape[4])

        targets = targets.to(opt.device)
        with torch.no_grad():
            inputs_visual = Variable(inputs_visual)
            inputs_audio = Variable(inputs_audio)
            targets = Variable(targets)
        '''
        outputs = model(inputs_audio, inputs_visual)
        loss = criterion(outputs, targets)
        '''
        # 更改后的多任务训练方法
        outputs = model(inputs_audio, inputs_visual)
        outputs_av = outputs['M']
        outputs_a = outputs['A']
        outputs_v = outputs['V']
        # outputs_Feature_a = outputs['Feature_a']  # torch.Size([8, 128])
        # outputs_Feature_v = outputs['Feature_v']  # torch.Size([8, 128])
        outputs_Feature_a_for_similar = outputs['Feature_a_for_similar']
        outputs_Feature_v_for_similar = outputs['Feature_v_for_similar']
        loss1 = criterion(outputs_av, targets)  # CrossEntropyLoss
        loss2 = criterion(outputs_a, targets)
        loss3 = criterion(outputs_v, targets)
        itcloss = BTwins(hidden_size=128, lambd=0, pj_size=128).cuda()
        loss4 = itcloss(outputs_Feature_a_for_similar, outputs_Feature_v_for_similar)
        # loss = 3*loss1 + loss2 + loss3 + 1e-6*loss4
        # loss = loss3
        loss = 3 * loss1 + loss2 + loss3
        prec1_vote, = calculate_accuracy_vote(torch.stack([outputs_av.data, outputs_a.data, outputs_v.data], dim=1),
                                             targets.data,topk=(1,))
        prec1, prec5 = calculate_accuracy(outputs_av.data, targets.data, topk=(1, 5))
        prec1_a, prec5_a = calculate_accuracy(outputs_a.data, targets.data, topk=(1, 5))
        prec1_v, prec5_v = calculate_accuracy(outputs_v.data, targets.data, topk=(1, 5))
        top1_vote.update(prec1_vote, inputs_audio.size(0))
        top1.update(prec1, inputs_audio.size(0))
        top5.update(prec5, inputs_audio.size(0))
        top1_a.update(prec1_a, inputs_audio.size(0))
        top5_a.update(prec5_a, inputs_audio.size(0))
        top1_v.update(prec1_v, inputs_audio.size(0))
        top5_v.update(prec5_v, inputs_audio.size(0))

        losses.update(loss.data, inputs_audio.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
              'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
              'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
            epoch,
            i + 1,
            len(data_loader),
            batch_time=batch_time,
            data_time=data_time,
            loss=losses,
            top1=top1,
            top5=top5))
        logging.debug('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
              'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
              'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
            epoch,
            i + 1,
            len(data_loader),
            batch_time=batch_time,
            data_time=data_time,
            loss=losses,
            top1=top1,
            top5=top5))
    #
    logger.log({'epoch': epoch,
                'loss': losses.avg.item(),
                'prec1': top1.avg.item(),
                'prec5': top5.avg.item()})
    wandb.log({'epoch': epoch,
                'val_loss': losses.avg.item(),
                'val_prec1_vote': top1_vote.avg.item(),
                'val_prec1': top1.avg.item(),
                'val_prec5': top5.avg.item(),
                'val_a_prec1': top1_a.avg.item(),
                'val_a_prec5': top5_a.avg.item(),
                'val_v_prec1': top1_v.avg.item(),
                'val_v_prec5': top5_v.avg.item(),
               })
    return losses.avg.item(), top1.avg.item()


def val_epoch(epoch, data_loader, model, criterion, opt, logger, modality='both', dist=None):
    print('validation at epoch {}'.format(epoch))
    logging.debug('validation at epoch {}'.format(epoch))
    if opt.model == 'multimodalcnn':
        return val_epoch_multimodal(epoch, data_loader, model, criterion, opt, logger, modality, dist=dist)
