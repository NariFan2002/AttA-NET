# -*- coding: utf-8 -*-
"""
Parts of this code are based on https://github.com/zengqunzhao/EfficientFace/blob/master/models/EfficientFace.py
"""

import torch
import torch.nn as nn
from models.modulator import Modulator
from models.efficientface import LocalFeatureExtractor, InvertedResidual
from models.transformer_timm import AttentionBlock, Attention
from models.WavLM import WavLM, WavLMConfig

def conv1d_block(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                         nn.BatchNorm1d(out_channels),
                         nn.ReLU(inplace=True))


class EfficientFaceTemporal(nn.Module):

    def __init__(self, stages_repeats, stages_out_channels, num_classes=7, im_per_sample=25):
        super(EfficientFaceTemporal, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(inplace=True), )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        self.local = LocalFeatureExtractor(29, 116, 1)
        self.modulator = Modulator(116)

        output_channels = self._stage_out_channels[-1]

        self.conv5 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(inplace=True), )
        self.conv1d_0 = conv1d_block(output_channels, 64)
        self.conv1d_1 = conv1d_block(64, 64)
        self.conv1d_2 = conv1d_block(64, 128)
        self.conv1d_3 = conv1d_block(128, 128)

        self.classifier_1 = nn.Sequential(
            nn.Linear(128, num_classes),
        )
        self.im_per_sample = im_per_sample

    def forward_features(self, x):

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.modulator(self.stage2(x)) + self.local(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # global average pooling
        # 增加的部分
        # assert x.shape[0] % self.im_per_sample == 0, "Batch size is not a multiple of sequence length."
        # n_samples = x.shape[0] // self.im_per_sample
        # x = x.view(n_samples, self.im_per_sample, x.shape[1])
        return x

    def forward_stage1(self, x):
        # Getting samples per batch

        assert x.shape[0] % self.im_per_sample == 0, "Batch size is not a multiple of sequence length."
        n_samples = x.shape[0] // self.im_per_sample
        x = x.view(n_samples, self.im_per_sample, x.shape[1])
        x = x.permute(0, 2, 1)
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        return x

    def forward_stage2(self, x):
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        return x

    def forward_classifier(self, x):
        x = x.mean([-1])  # pooling accross temporal dimension
        x1 = self.classifier_1(x)
        return x1

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_stage1(x)
        x = self.forward_stage2(x)
        x = self.forward_classifier(x)
        return x


def init_feature_extractor(model, path):
    if path == 'None' or path is None:
        return
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    pre_trained_dict = checkpoint['state_dict']
    pre_trained_dict = {key.replace("module.", ""): value for key, value in pre_trained_dict.items()}
    print('Initializing efficientnet')
    model.load_state_dict(pre_trained_dict, strict=False)


def get_model(num_classes, task, seq_length):
    model = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes, task, seq_length)
    return model


def conv1d_block_audio(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding='valid'),
                         nn.BatchNorm1d(out_channels),
                         nn.ReLU(inplace=True), nn.MaxPool1d(2, 1))


class AudioCNNPool(nn.Module):

    def __init__(self, num_classes=8):
        super(AudioCNNPool, self).__init__()

        input_channels = 10
        self.conv1d_0 = conv1d_block_audio(input_channels, 64)
        self.conv1d_1 = conv1d_block_audio(64, 128)
        self.conv1d_2 = conv1d_block_audio(128, 256)
        self.conv1d_3 = conv1d_block_audio(256, 128)

        self.classifier_1 = nn.Sequential(
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.forward_stage1(x)
        x = self.forward_stage2(x)
        x = self.forward_classifier(x)
        return x

    def forward_stage1(self, x):
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        return x

    def forward_stage2(self, x):
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        return x

    def forward_classifier(self, x):
        x = x.mean([-1])  # pooling accross temporal dimension
        x1 = self.classifier_1(x)
        return x1


class MultiModalCNN(nn.Module):
    # def __init__(self, num_classes=8, fusion='ia', seq_length=15, pretr_ef='None', num_heads=1):
    #     super(MultiModalCNN, self).__init__()
    #     assert fusion in ['ia', 'it', 'lt', 'lt_wo_attention_block'], print('Unsupported fusion method: {}'.format(fusion))
    #     # 初始化语音特征提取模型
    #     checkpoint = torch.load('/home/fanrj/pythonproject/multimodal-emotion-recognition-AV-RAVDESS/models/WavLM-Large.pt')
    #     cfg = WavLMConfig(checkpoint['cfg'])
    #     self.audio_model_WavLM = WavLM(cfg)
    #     self.audio_model_WavLM.load_state_dict(checkpoint['model'])
    #     self.relu = torch.nn.ReLU()
    #     for p in self.audio_model_WavLM.parameters():
    #         p.requires_grad = False
    #     self.audio_prefeature_norm1 = nn.Linear(in_features=1024,out_features=128)
    #     self.audio_prefeature_norm2 = nn.Linear(in_features=128, out_features=8)
    #     self.audio_model = AudioCNNPool(num_classes=num_classes)
    #     # 初始化视频特征提取模型
    #     self.visual_model = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes, seq_length)
    #     # 初始化视频模型参数
    #     init_feature_extractor(self.visual_model, pretr_ef)
    #
    #     e_dim = 128
    #     input_dim_video = 128
    #     input_dim_audio = 128
    #     self.fusion = fusion
    #
    #     if fusion in ['lt', 'it']:
    #         if fusion == 'lt':
    #             self.av = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=e_dim,
    #                                      num_heads=num_heads)
    #             self.va = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=e_dim,
    #                                      num_heads=num_heads)
    #             self.vv = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=e_dim,
    #                                      num_heads=num_heads)
    #             self.aa = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=e_dim,
    #                                      num_heads=num_heads)
    #         elif fusion == 'it':
    #             input_dim_video = input_dim_video // 2
    #             self.av1 = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio,
    #                                       num_heads=num_heads)
    #             self.va1 = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video,
    #                                       num_heads=num_heads)
    #
    #     elif fusion in ['ia']:
    #         input_dim_video = input_dim_video // 2
    #
    #         self.av1 = Attention(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio,
    #                              num_heads=num_heads)
    #         self.va1 = Attention(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video,
    #                              num_heads=num_heads)
    #     # 使用多模态concat后的特征来分类
    #     self.post_fusion_dropout = nn.Dropout(p=0.5)
    #     self.classifier_1_norelu = nn.Sequential(
    #         nn.Linear(e_dim * 2, 32),
    #         nn.Linear(32, num_classes)
    #     )
    #     self.classifier_2_norelu = nn.Sequential(
    #         nn.Linear(input_dim_audio, 64),
    #         nn.Linear(64, 32),
    #         nn.Linear(32, num_classes)
    #     )
    #     self.classifier_3_norelu = nn.Sequential(
    #         nn.Linear(input_dim_audio, 64),
    #         nn.Linear(64, 32),
    #         nn.Linear(32, num_classes)
    #     )
    #
    #
    #     self.classifier_1 = nn.Sequential(
    #         nn.Linear(e_dim * 2, 32),
    #         nn.ReLU(),
    #         nn.Linear(32, num_classes)
    #     )
    #     # 设定仅仅只使用音频的特征来分类 the classify layer for audio
    #     self.post_audio_dropout = nn.Dropout(p=0.5)
    #     self.classifier_2 = nn.Sequential(
    #         nn.Linear(input_dim_audio, 256),
    #         nn.ReLU(),
    #         nn.Linear(256, 64),
    #         nn.ReLU(),
    #         nn.Linear(64, num_classes)
    #     )
    #     # 设定只使用视频的特征来分类 the classify layer for video
    #     self.post_video_dropout = nn.Dropout(p=0.5)
    #     self.classifier_3 = nn.Sequential(
    #         nn.Linear(input_dim_audio, 256),
    #         nn.ReLU(),
    #         nn.Linear(256, 64),
    #         nn.ReLU(),
    #         nn.Linear(64, num_classes)
    #     )
    def __init__(self, num_classes=8, fusion='ia', seq_length=15, pretr_ef='None', num_heads=1):
        super(MultiModalCNN, self).__init__()
        assert fusion in ['ia', 'it', 'lt', 'lt_wo_attention_block'], print('Unsupported fusion method: {}'.format(fusion))
        self.audio_model = AudioCNNPool(num_classes=num_classes)
        # 初始化视频特征提取模型
        self.visual_model = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes, seq_length)
        # 初始化视频模型参数
        init_feature_extractor(self.visual_model, pretr_ef)

        e_dim = 128
        input_dim_video = 128
        input_dim_audio = 128
        self.fusion = fusion

        if fusion in ['lt', 'it']:
            if fusion == 'lt':
                self.av = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=e_dim,
                                         num_heads=num_heads)
                self.va = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=e_dim,
                                         num_heads=num_heads)
                self.vv = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=e_dim,
                                         num_heads=num_heads)
                self.aa = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=e_dim,
                                         num_heads=num_heads)
            elif fusion == 'it':
                input_dim_video = input_dim_video // 2
                self.av1 = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio,
                                          num_heads=num_heads)
                self.va1 = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video,
                                          num_heads=num_heads)

        elif fusion in ['ia']:
            input_dim_video = input_dim_video // 2

            self.av1 = Attention(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio,
                                 num_heads=num_heads)
            self.va1 = Attention(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video,
                                 num_heads=num_heads)
        # 使用多模态concat后的特征来分类
        self.post_fusion_dropout = nn.Dropout(p=0.5)
        self.post_audio_dropout = nn.Dropout(p=0.5)
        self.post_video_dropout = nn.Dropout(p=0.5)
        self.classifier_1 = nn.Sequential(
            nn.Linear(e_dim * 2, 32),
            nn.Linear(32, num_classes)
        )
        self.classifier_2 = nn.Sequential(
            nn.Linear(input_dim_audio, 64),
            nn.Linear(64, 32),
            nn.Linear(32, num_classes)
        )
        self.classifier_3 = nn.Sequential(
            nn.Linear(input_dim_audio, 64),
            nn.Linear(64, 32),
            nn.Linear(32, num_classes)
        )





    def forward(self, x_audio, x_visual):
        if self.fusion == 'lt':
            # return self.forward_transformer(x_audio, x_visual)
            return self.forward_transformer_compress(x_audio, x_visual)
            # 明天来看一下大模型的表现能力
            # return self.forward_audio(x_audio,x_visual)
        elif self.fusion == 'ia':
            # return self.forward_feature_2(x_audio, x_visual)
            return self.forward_audio(x_audio, x_visual)
        elif self.fusion == 'it':
            return self.forward_feature_3(x_audio, x_visual)
        elif self.fusion == 'lt_wo_attention_block':
            return self.forward_transformer_compress_wo_attention_block(x_audio, x_visual)

    def forward_audio(self, x_audio,x_video):
        x_audio = self.audio_model_WavLM.extract_features(x_audio)[0].mean([1])
        proj_x_a = self.audio_prefeature_norm1(x_audio)
        x2 = self.audio_prefeature_norm2(proj_x_a)
        # print(x2.shape)
        ret = {
            'M': x2,
            'A': x2,
            'V': x2,
        }
        return ret


    def forward_feature_3(self, x_audio, x_visual):
        x_audio = self.audio_model.forward_stage1(x_audio)
        x_visual = self.visual_model.forward_features(x_visual)
        x_visual = self.visual_model.forward_stage1(x_visual)

        proj_x_a = x_audio.permute(0, 2, 1)
        proj_x_v = x_visual.permute(0, 2, 1)

        h_av = self.av1(proj_x_v, proj_x_a)
        h_va = self.va1(proj_x_a, proj_x_v)

        h_av = h_av.permute(0, 2, 1)
        h_va = h_va.permute(0, 2, 1)

        x_audio = h_av + x_audio
        x_visual = h_va + x_visual

        x_audio = self.audio_model.forward_stage2(x_audio)
        x_visual = self.visual_model.forward_stage2(x_visual)

        audio_pooled = x_audio.mean([-1])  # mean accross temporal dimension
        video_pooled = x_visual.mean([-1])

        x = torch.cat((audio_pooled, video_pooled), dim=-1)
        x1 = self.classifier_1(x)
        return x1

    def forward_feature_2(self, x_audio, x_visual):
        x_audio = self.audio_model.forward_stage1(x_audio)
        x_visual = self.visual_model.forward_features(x_visual)
        x_visual = self.visual_model.forward_stage1(x_visual)

        proj_x_a = x_audio.permute(0, 2, 1)
        proj_x_v = x_visual.permute(0, 2, 1)

        _, h_av = self.av1(proj_x_v, proj_x_a)
        _, h_va = self.va1(proj_x_a, proj_x_v)

        if h_av.size(1) > 1:  # if more than 1 head, take average
            h_av = torch.mean(h_av, axis=1).unsqueeze(1)

        h_av = h_av.sum([-2])

        if h_va.size(1) > 1:  # if more than 1 head, take average
            h_va = torch.mean(h_va, axis=1).unsqueeze(1)

        h_va = h_va.sum([-2])

        x_audio = h_va * x_audio
        x_visual = h_av * x_visual

        x_audio = self.audio_model.forward_stage2(x_audio)
        x_visual = self.visual_model.forward_stage2(x_visual)

        audio_pooled = x_audio.mean([-1])  # mean accross temporal dimension
        video_pooled = x_visual.mean([-1])

        x = torch.cat((audio_pooled, video_pooled), dim=-1)

        x1 = self.classifier_1(x)
        return x1

    def forward_transformer(self, x_audio, x_visual):
        x_audio = self.audio_model.forward_stage1(x_audio)
        proj_x_a = self.audio_model.forward_stage2(x_audio)

        x_visual = self.visual_model.forward_features(x_visual)
        x_visual = self.visual_model.forward_stage1(x_visual)
        proj_x_v = self.visual_model.forward_stage2(x_visual)

        proj_x_a = proj_x_a.permute(0, 2, 1)
        proj_x_v = proj_x_v.permute(0, 2, 1)
        h_av = self.av(proj_x_v, proj_x_a)
        h_va = self.va(proj_x_a, proj_x_v)

        audio_pooled = h_av.mean([1])  # mean accross temporal dimension
        video_pooled = h_va.mean([1])

        # fusion feature
        x = torch.cat((audio_pooled, video_pooled), dim=-1)
        x = self.post_fusion_dropout(x)

        # classifier-fusion
        x1 = self.classifier_1(x)
        return x1

    def forward_transformer_compress_large(self, x_audio, x_visual):
        # print(' 进入推倒过程：forward_transformer_compress')
        # x_audio = self.audio_model.forward_stage1(x_audio)
        # proj_x_a = self.audio_model.forward_stage2(x_audio)
        # x_audio = self.audio_model_WavLM.extract_features(x_audio)[0]
        # print('x_audio.shape',x_audio.shape)
        # batchsize * 179 * 1024
        # print('音频特征提取后的要求a:', x_audio.shape)
        x_visual = self.visual_model.forward_features(x_visual)
        # print('视频特征提取后的要求v:', x_visual.shape)
        proj_x_a = x_audio
        proj_x_v = x_visual
        # print('自注意力提纯前的要求v,a:',proj_x_v.shape,proj_x_a.shape)
        # 先进行自注意力提纯information
        # proj_x_v = self.vv(proj_x_v,proj_x_v)
        proj_x_a = self.aa(proj_x_a, proj_x_a)
        # print('自注意力提纯后的要求v,a:', proj_x_v.shape, proj_x_a.shape)
        # 进行模型压缩
        h_av = self.av(proj_x_v, proj_x_a)
        h_va = self.av(proj_x_a, proj_x_v)
        # print('模型压缩av,va:',h_av.shape,h_va.shape)
        audio_pooled = h_av.mean([1])  # mean accross temporal dimension
        video_pooled = h_va.mean([1])

        # fusion feature
        x = torch.cat((audio_pooled, video_pooled), dim=-1)
        x = self.post_fusion_dropout(x)
        x_audio = self.post_audio_dropout(proj_x_a.mean([1]))
        x_visual = self.post_video_dropout(proj_x_v.mean([1]))
        # classifier-fusion
        x1 = self.classifier_1(x)
        x2 = self.classifier_2(x_audio)
        x3 = self.classifier_3(x_visual)

        ret = {
            'M':x1,
            'A':x2,
            'V':x3,
            'Feature_m': x,
            'Feature_a':x_audio,
            'Feature_v':x_visual
        }
        # print('ret',x1.shape,x2.shape,x3.shape,x.shape,x_audio.shape,x_visual.shape)
        return ret

    def forward_transformer_compress(self, x_audio, x_visual):
        # print(' 进入推倒过程：forward_transformer_compress')
        x_audio = self.audio_model.forward_stage1(x_audio)
        proj_x_a = self.audio_model.forward_stage2(x_audio)
        x_visual = self.visual_model.forward_features(x_visual)
        x_visual = self.visual_model.forward_stage1(x_visual)
        proj_x_v = self.visual_model.forward_stage2(x_visual)

        proj_x_a = proj_x_a.permute(0, 2, 1)
        proj_x_v = proj_x_v.permute(0, 2, 1)

        proj_x_a = self.aa(proj_x_a, proj_x_a)

        h_av = self.av(proj_x_v, proj_x_a) # 视频分支
        h_va = self.av(proj_x_a, proj_x_v) # 音频分支

        # h_av_res = h_av+proj_x_v #视频分支进行操作
        # h_va_res = h_va+proj_x_a #音频分支进行操作
        audio_pooled = h_av.mean([1])  # mean accross temporal dimension
        video_pooled = h_va.mean([1])

        # fusion feature

        x_audio = self.post_audio_dropout(proj_x_a.mean([1]))
        x_visual = self.post_video_dropout(proj_x_v.mean([1]))
        # x = torch.cat((audio_pooled, video_pooled), dim=-1)
        x = torch.cat((audio_pooled+x_audio, video_pooled+x_visual), dim=-1)
        x = self.post_fusion_dropout(x)

        # classifier-fusion
        x1 = self.classifier_1(x)
        x2 = self.classifier_2(x_audio)
        x3 = self.classifier_3(x_visual)

        ret = {
            'M':x1,
            'A':x2,
            'V':x3,
            'Feature_m': x,
            'Feature_a':x_audio,
            'Feature_v':x_visual,
            'Feature_a_for_similar': audio_pooled,
            'Feature_v_for_similar': video_pooled,
        }
        # print('ret',x1.shape,x2.shape,x3.shape,x.shape,x_audio.shape,x_visual.shape)
        return ret

    def forward_transformer_compress_wo_attention_block(self, x_audio, x_visual):
        # print(' 进入推倒过程：forward_transformer_compress')
        x_audio = self.audio_model.forward_stage1(x_audio)
        proj_x_a = self.audio_model.forward_stage2(x_audio)

        x_visual = self.visual_model.forward_features(x_visual)
        x_visual = self.visual_model.forward_stage1(x_visual)
        proj_x_v = self.visual_model.forward_stage2(x_visual)

        proj_x_a = proj_x_a.permute(0, 2, 1)
        proj_x_v = proj_x_v.permute(0, 2, 1)
        '''
        # 将attention block部分进行注释
        proj_x_a = self.aa(proj_x_a, proj_x_a)
        # print('自注意力提纯后的要求v,a:', proj_x_v.shape, proj_x_a.shape)
        # 进行模型压缩
        h_av = self.av(proj_x_v, proj_x_a)
        h_va = self.av(proj_x_a, proj_x_v)
        # print('模型压缩av,va:',h_av.shape,h_va.shape)
        '''
        h_av = proj_x_a
        h_va = proj_x_v
        audio_pooled = h_av.mean([1])  # mean accross temporal dimension
        video_pooled = h_va.mean([1])
        # fusion feature


        x_audio = self.post_audio_dropout(proj_x_a.mean([1]))
        x_visual = self.post_video_dropout(proj_x_v.mean([1]))

        x = torch.cat((audio_pooled+x_audio, video_pooled+x_visual), dim=-1)
        x = self.post_fusion_dropout(x)
        # classifier-fusion
        x1 = self.classifier_1(x)
        x2 = self.classifier_2(x_audio)
        x3 = self.classifier_3(x_visual)
        # output
        ret = {
            'M':x1,
            'A':x2,
            'V':x3,
            'Feature_m': x,
            'Feature_a':x_audio,
            'Feature_v':x_visual,
            'Feature_a_for_similar':audio_pooled,
            'Feature_v_for_similar': video_pooled,
        }
        return ret

















