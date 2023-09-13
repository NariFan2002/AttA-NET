'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''

from torch import nn

from models import multimodalcnn


def generate_model(opt):
    assert opt.model in ['multimodalcnn']



    if opt.model == 'multimodalcnn':
        model = multimodalcnn.MultiModalCNN(opt.n_classes, fusion=opt.fusion, seq_length=opt.sample_duration,
                                            pretr_ef=opt.pretrain_path, num_heads=opt.num_heads)

    if opt.device != 'cpu':
        model = model.to(opt.device) #将模型放到设备上面
        model = nn.DataParallel(model, device_ids=None)
        '''
        module (Module) – module to be parallelized 
        device_ids (list of python:int or torch.device) – CUDA devices (default: all devices)
        output_device (int or torch.device) – device location of output (default: device_ids[0])
        '''

        pytorch_total_params = sum(p.numel() for p in model.parameters() if
                                   p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)
        pytorch_total_params_ = sum(p.numel() for p in model.parameters())
        print("Total number of parameters: ", pytorch_total_params_)
    return model, model.parameters()