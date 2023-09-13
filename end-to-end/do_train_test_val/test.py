'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''
import torch
from torch.autograd import Variable
import time
from utils import AverageMeter, calculate_accuracy, calculate_accuracy_vote, take_the_output_mean
import logging
import wandb
from .draw_tsne import draw_tsne

def test_epoch_multimodal(epoch, data_loader, model, criterion, opt, logger, modality='both', dist=None):
    # for evaluation with single modality, specify which modality to keep and which distortion to apply for the other modaltiy:
    # 'noise', 'addnoise' or 'zeros'. for paper procedure, with 'softhard' mask use 'zeros' for evaluation, with 'noise' use 'noise'
    print('test at epoch {}'.format(epoch))
    logging.debug('test at epoch {}'.format(epoch) + 'or test when epoch == 10000')
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

    # 用来存储特征的情况
    Feature_map={
        'fusion':torch.zeros(opt.test_samples, opt.post_fusion_dim, requires_grad=False).to(opt.device),
        'audio': torch.zeros(opt.test_samples, opt.post_audio_dim, requires_grad=False).to(opt.device),
        'vision': torch.zeros(opt.test_samples, opt.post_video_dim, requires_grad=False).to(opt.device),
    }
    Target_map = torch.zeros(opt.test_samples,dtype=int, requires_grad=False)
    Prediction_map_m = torch.zeros(opt.test_samples,dtype=int, requires_grad=False).to(opt.device)
    Prediction_map_a = torch.zeros(opt.test_samples,dtype=int, requires_grad=False).to(opt.device)
    Prediction_map_v = torch.zeros(opt.test_samples,dtype=int, requires_grad=False).to(opt.device)

    end_time = time.time()
    for i, (inputs_audio, inputs_visual, targets, id) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        # print("Target_map.is_cuda:",Target_map.is_cuda," id.is_cuda:",id.is_cuda," targets.is_cuda:",targets.is_cuda)
        Target_map[id] = targets
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
        # print(inputs_visual.is_cuda,inputs_audio.is_cuda)
        outputs = model(inputs_audio, inputs_visual)
        outputs_av = outputs['M']
        outputs_a = outputs['A']
        outputs_v = outputs['V']
        # outputs_av = take_the_output_mean(outputs_av,outputs_a,outputs_v,0,0,3)

        f_fusion = outputs['Feature_m'].detach()
        f_audio = outputs['Feature_a'].detach()
        f_vision = outputs['Feature_v'].detach()
        # print('f_fusion.is_cuda', f_fusion.is_cuda)
        loss1 = criterion(outputs_av, targets)  # CrossEntropyLoss
        loss2 = criterion(outputs_a, targets)
        loss3 = criterion(outputs_v, targets)
        loss = 3*loss1+loss2+loss3
        # 输出预测的结果看一看
        # loss = loss1
        # print('输出预测的结果看一看')
        # print('outputs_av:',outputs_av.shape,outputs_av[0])
        # print('outputs_v:',outputs_v.shape,outputs_v[0])
        # print('outputs_a:',outputs_a.shape,outputs_a[0])

        Feature_map = update_features(Feature_map,f_fusion, f_audio, f_vision, id)
        Prediction_map_v = update_Predicttions(Prediction_map_v,outputs_v.argmax(dim=1),id)
        Prediction_map_a = update_Predicttions(Prediction_map_a, outputs_a.argmax(dim=1), id)
        Prediction_map_m = update_Predicttions(Prediction_map_m, outputs_av.argmax(dim=1), id)

        print("-----输出目标值和预测值------")
        print("id:",id)
        print("av判断结果：")
        prec1, prec5 = calculate_accuracy(outputs_av.data, targets.data, topk=(1, 5))
        prec1_vote, = calculate_accuracy_vote(torch.stack([outputs_av.data,outputs_a.data,outputs_v.data],dim=1),targets.data,topk=(1,))
        print("a判断结果：")
        prec1_a, prec5_a = calculate_accuracy(outputs_a.data, targets.data, topk=(1, 5))
        print("v判断结果：")
        prec1_v, prec5_v = calculate_accuracy(outputs_v.data, targets.data, topk=(1, 5))
        print("----结束盖batch的输出-------")
        top1.update(prec1, inputs_audio.size(0))
        top1_vote.update(prec1_vote, inputs_audio.size(0))
        top5.update(prec5, inputs_audio.size(0))
        top1_a.update(prec1_a, inputs_audio.size(0))
        top5_a.update(prec5_a, inputs_audio.size(0))
        top1_v.update(prec1_v, inputs_audio.size(0))
        top5_v.update(prec5_v, inputs_audio.size(0))

        losses.update(loss.data, inputs_audio.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

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

    # print('Feature_map',Feature_map)
    # print('Feature_map[fusion]:',Feature_map['fusion'].shape)
    # print('Target_map',Target_map)
    # print('Feature_map.is_cuda:',Feature_map['fusion'].is_cuda,'Target_map.is_cuda',Target_map.is_cuda)
    # torch.save(Feature_map['fusion'].cpu(),'Feature_map[fusion].t')
    # torch.save(Target_map.cpu(),'Target_map.t')
    # print('将特征和target保存好')

    fig_m = draw_tsne(Feature_map['fusion'].cpu(), Target_map.cpu(), Prediction_map_m.cpu())
    fig_a = draw_tsne(Feature_map['audio'].cpu(), Target_map.cpu(), Prediction_map_a.cpu())
    fig_v = draw_tsne(Feature_map['vision'].cpu(), Target_map.cpu(), Prediction_map_v.cpu())
    image_m = wandb.Image(fig_m, caption=f"测试集融合特征的分类图片")
    image_a = wandb.Image(fig_a, caption=f"测试集a特征的分类图片")
    iamge_v = wandb.Image(fig_v, caption=f"测试集v特征的分类图片")
    wandb.log({'image_m':image_m,"image_a":image_a,"image_v":iamge_v})

    logger.log({'epoch': epoch,
                'loss': losses.avg.item(),
                'prec1': top1.avg.item(),
                'prec5': top5.avg.item()})
    wandb.log({ 'epoch': epoch,
                'test_loss': losses.avg.item(),
                'test_prec1_vote': top1_vote.avg.item(),
                'test_prec1': top1.avg.item(),
                'test_prec5': top5.avg.item(),
                'test_a_prec1': top1_a.avg.item(),
                'test_a_prec5': top5_a.avg.item(),
                'test_v_prec1': top1_v.avg.item(),
                'test_v_prec5': top5_v.avg.item()}
              )
    return losses.avg.item(), top1.avg.item()

def update_features(feature_map, f_fusion, f_audio, f_vision, id):
    feature_map['fusion'][id] = f_fusion
    feature_map['audio'][id] = f_audio
    feature_map['vision'][id] = f_vision
    return feature_map

def update_Predicttions(Prediction,output,id):
    Prediction[id] = output
    return Prediction

def test_epoch(epoch, data_loader, model, criterion, opt, logger, modality='both', dist=None):
    print('test at epoch {}'.format(epoch))
    logging.debug('test at epoch {}'.format(epoch))
    if opt.model == 'multimodalcnn':
        return test_epoch_multimodal(epoch, data_loader, model, criterion, opt, logger, modality, dist=dist)
