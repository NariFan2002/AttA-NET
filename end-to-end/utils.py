'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''
import csv
import torch
import shutil
import numpy as np
import sklearn
import wandb
from librosa import ex


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()
def most_frequent(tensor):
    if tensor.numel() == 0:
        return None
    counts = torch.bincount(tensor.view(-1))
    max_count = counts.max()
    if (counts == max_count).sum() > 1:
        return None
    return torch.argmax(counts).item()
def calculate_accuracy_vote(output,target,topk=(1,), binary=False):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    if maxk > output.size(2):
        maxk = output.size(2)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 2, True, True)
    pred = pred.transpose(1, 2)

    correct = torch.zeros((batch_size, maxk), dtype=torch.bool)
    for i in range(batch_size):
        for j in range(maxk):
            if most_frequent(pred[i]):
                correct[i, j] = (most_frequent(pred[i, :, j]) == target[i])
                break
    # print('输出布尔值')
    # print(correct)
    res = []
    for k in topk:
        if k > maxk:
            k = maxk
        correct_k = correct[:, :k].reshape(-1).float().sum(0)
        # print(correct_k)
        res.append(correct_k.mul_(100.0 / batch_size))
    if binary:
        f1 = sklearn.metrics.f1_score(list(target.cpu().numpy()), list(pred[:, 0, 0].cpu().numpy()))
        return res, f1 * 100
    # print('res',res)
    return res

def calculate_accuracy(output, target, topk=(1,), binary=False):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    # print('target', target, 'output', output)
    if maxk > output.size(1):
        maxk = output.size(1)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    print('Target: ', target, 'Pred: ', pred)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        if k > maxk:
            k = maxk
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    if binary:
        # print(list(target.cpu().numpy()),  list(pred[0].cpu().numpy()))
        f1 = sklearn.metrics.f1_score(list(target.cpu().numpy()), list(pred[0].cpu().numpy()))
        # print('F1: ', f1)
        return res, f1 * 100
    # print('no_vote_res:',res)
    return res

def take_the_output_mean(tensor1,tensor2,tensor3,lambda1,lambda2,lambda3):
    # 对每个 tensor 进行归一化
    from torch.nn.functional import normalize
    norm1 = normalize(tensor1, dim=1)
    norm2 = normalize(tensor2, dim=1)
    norm3 = normalize(tensor3, dim=1)
    result = lambda1*norm1+lambda2*norm2+lambda3*norm3
    return result


def save_checkpoint(state, is_best, opt, fold):
    torch.save(state, '%s/%s_checkpoint' % (opt.result_path, opt.store_name) + str(fold) + '.pth')
    if is_best:
        shutil.copyfile('%s/%s_checkpoint' % (opt.result_path, opt.store_name) + str(fold) + '.pth',
                        '%s/%s_best' % (opt.result_path, opt.store_name) + str(fold) + '.pth')


def adjust_learning_rate(optimizer, epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs
        将学习率设置为初始LR每30次衰减10
    """
    lr_new = opt.learning_rate * (0.1 ** (sum(epoch >= np.array(opt.lr_steps))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
        # param_group['lr'] = opt.learning_rate

from torch  import nn
# 计算ict loss
class BTwins(nn.Module):
    def __init__(self, hidden_size, lambd, pj_size):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, pj_size, bias=False),
            nn.BatchNorm1d(pj_size),
            nn.ReLU(True),
            nn.Linear(pj_size, pj_size, bias=False),
            nn.BatchNorm1d(pj_size),
            nn.ReLU(True),
            nn.Linear(pj_size, pj_size, bias=False),
        )
        self.bn = nn.BatchNorm1d(pj_size, affine=False)
        self.lambd = lambd
        self.projector1 = nn.Linear(hidden_size,8)

    def forward(self, feat1, feat2):
        feat1 = self.projector(feat1)
        feat2 = self.projector(feat2)
        feat1_norm = self.bn(feat1)
        feat2_norm = self.bn(feat2)

        N, D = feat1_norm.shape
        c = (feat1_norm.T @ feat2_norm).div_(N)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        BTloss = on_diag + self.lambd * off_diag

        return BTloss

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()