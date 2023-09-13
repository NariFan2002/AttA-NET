"""
    Script to emotion recognition

	author: Ruijia Fan.
	date: 09/2023

	Usage:
		e.g.
		 python3 MMEmotionRecognition/src/Audio/FeatureExtractionWav2Vec/FeatureExtractor.py
		 --data MMEmotionRecognition/data/models/wav2Vec_top_models/FineTuning/data/20211020_094500
		 --model_id jonatasgrosman/wav2vec2-large-xlsr-53-english
		 --out_dir <RAVDESS_dir>/FineTuningWav2Vec2_embs512
	Options:
        --data: Path with the datasets automatically generated with the Fine-Tuning script (train.csv and test.csv)
		--model_id: Path to the baseline model to extract the features from
		--out_dir: Path to save the embeddings
"""
import os.path, os, sys
import argparse
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import wandb
import pandas as pd
import numpy as np
from src.Audio.FeatureExtractionWav2Vec.FeatureTraining import get_classifier, extract_posteriors
import sklearn
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from  torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from src.Fusion.model.transformer_timm import AttentionBlock,Attention
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

class RAVDESSDataset(Dataset):
    def __init__(self,x,y):
        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y.values)
    def __getitem__(self, item):
        return torch.DoubleTensor(self.x_data[item]), self.y_data[item]
    def __len__(self):
        return self.len

class MultiModalCNN(nn.Module):
    def __init__(self, num_classes=8,num_heads=1):
        super(MultiModalCNN, self).__init__()
        input_dim_video = 32
        input_dim_audio = 32
        e_dim = 64
        # input_dim_video = input_dim_video // 2
        self.audio_project_Linear = nn.Linear(8, 32).double()
        self.video_project_Linear = nn.Linear(16, 32).double()
        self.aa = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=e_dim,
                                 num_heads=num_heads).double()
        self.av = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=e_dim,
                                 num_heads=num_heads).double()
        self.classifier_1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(e_dim * 2, 32).double(),
            nn.ReLU(),
            nn.Linear(32, num_classes).double()
        )
        self.classifier_2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(e_dim, num_classes).double(),
        )
        self.classifier_3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(e_dim, num_classes).double(),
        )
    def forward(self,audio,video):
        # audio = self.aa(audio,audio)
        audio = audio.reshape(-1,1,8)
        video = video.reshape(-1,1,16)
        # print(audio.shape,audio.dtype)
        # print(video.shape)
        # print(self.audio_project_Linear.weight.dtype)
        audio = self.audio_project_Linear(audio)
        video = self.video_project_Linear(video)

        proj_video = self.av(audio,video)
        proj_audio = self.av(video,audio)
        a = self.classifier_2(proj_audio)
        v = self.classifier_3(proj_video)
        # print(proj_audio.shape, audio.dtype)
        # print(proj_video.shape)
        m = torch.concat((proj_audio,proj_video),dim=2)
        # print(m.shape)
        m = self.classifier_1(m)
        m = m.reshape(-1,8)
        a = a.reshape(-1,8)
        v = v.reshape(-1,8)
        return m,a,v
def calculate_accuracy(output, target, topk=(1,), binary=False):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    # print('target', target, 'output', output)
    if maxk > output.size(1):
        maxk = output.size(1)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # print('Target: ', target, 'Pred: ', pred)
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

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1) # only difference

def remove_cols(df, cols2rm=[]):
    for col in cols2rm:
        try:
            df = df.drop(columns=[col])
        except KeyError:
            continue
    return df


def prepare_video_modality(X_total, fold):
    actors_per_fold = {
        0: [2, 5, 14, 15, 16],
        1: [3, 6, 7, 13, 18],
        2: [10, 11, 12, 19, 20],
        3: [8, 17, 21, 23, 24],
        4: [1, 4, 9, 22],
    }
    X_total["actor"] +=1

    test_df = X_total.loc[X_total['actor'].isin(actors_per_fold[fold])]
    train_df = X_total.loc[~X_total['actor'].isin(actors_per_fold[fold])]

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df

def prepare_video_modality_biLSTMAtt(df):
    #Convert emotions:
    df["emotion"] = df["tag"].str.split("-").str[2].astype(int)
    df["emotion"] -= 1
    #Extend posteriors columns
    df[["posterios"+str(i) for i in range(8)]] = df["posteriors"].str.split("[").str[-1].str.split("]").str[0].str.split(",", expand=True)
    df[["posterios"+str(i) for i in range(8)]] = df[["posterios"+str(i) for i in range(8)]].apply(pd.to_numeric)
    df["tag"] = df["tag"].str.replace(".csv", "")
    #remove cols:
    df = remove_cols(df, cols2rm=[ "y", "y_hat", "posteriors", "attentions"])
    df = df.rename(columns = {"tag":"name"})
    return df



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-embsWav2vec', '--embs_dir_wav2vec',default='/home/fanrj/pythonproject/MMEmotionRecognition/data/posteriors/wav2Vec/posteriors/20211020_094500', type=str,
                        help='Path with the embeddings of the fine-tuned Wav2Vec model')
    parser.add_argument('-embsBiLSTM', '--embs_dir_biLSTM', default='/home/fanrj/pythonproject/MMEmotionRecognition/data/posteriors/AUs_biLSTM_6213/posteriorsv2', type=str,
                        help='Path with the embeddings of the bi-LSTM with attention mechanism trained with the AUs')
    parser.add_argument('-embsMLP', '--embs_dir_MLP',default='/home/fanrj/pythonproject/MMEmotionRecognition/data/posteriors/avg_MLP80_AUs/posteriors', type=str,
                        help='Path with the embeddings of the MLP trained with the average of the AUs')
    parser.add_argument('-out', '--out_dir', type=str, help='Path to save the embeddings extracted from the model [default: Not save embs]',
                        default='')
    parser.add_argument('-m', '--model_number', type=int,
                        help='1-SVC / 2- Logistic Regression / 3- ridgeClassifier /4-perceptron / 5-NuSVC / 6-LinearSVC / 7-knn / 8-NearestCentroid / 9- DecrissionTree / 10- RandomForest / 11 - MLP',default=2)
    parser.add_argument('-modelParam', '--param', type=str,
                        help='Parameter of the model: C for SVC / C Logistic Regression / alpha in ridgeClassifier / alpha in perceptron / nu in NuSVC / C in LinearSVC / k in knn / None in NearestCentroid / min_samples_split in DecrissionTree / n_estimators in RandomForest / hidden_layer_sizes in MLP',
                        default=1.0)
    parser.add_argument('-norm', '--type_of_norm', type=int,
                        help='0-MinMax Norm / 1-Standard Norm / 2- No apply normalization [default: 2]', default=1)

    args = parser.parse_args()
    # print(args)
    seed = 2020

    if args.out_dir=='':
        get_embs = False
    else:
        get_embs = True
        os.makedirs(args.out_dir, exist_ok=True)

    wandb.init(
        # set the wandb project where this run will be logged
        project="EEmotion",
        name='1',
        # track hyperparameters and run metadata
        # config=vars()  # vars将opt的Namespace object变成dict #保存网络参数信息
    )
    avg_acc = 0
    for fold in range(1):
        # AUDIO WAV2VEC MODEL
        X_train_audio = pd.read_csv(os.path.join(args.embs_dir_wav2vec, "fold" + str(fold), "posteriors_train.csv"), sep=";",
                                    header=0)
        X_test_audio = pd.read_csv(os.path.join(args.embs_dir_wav2vec, "fold" + str(fold), "posteriors_test.csv"), sep=";",
                                   header=0)

        X_train_audio = X_train_audio.sort_values(by=["name"])
        X_train_audio = X_train_audio.reset_index(drop=True)

        X_test_audio = X_test_audio.sort_values(by=["name"])
        X_test_audio = X_test_audio.reset_index(drop=True)

        # VIDEO SEQUENTIAL MODELS
        X_train_video = pd.read_csv(
            os.path.join(args.embs_dir_biLSTM, "RAVDESS_AUs-train--fold_" + str(fold + 1) + "_outof_5.csv"), sep="\t",
            header=0)
        X_test_video = pd.read_csv(
            os.path.join(args.embs_dir_biLSTM, "RAVDESS_AUs-val--fold_" + str(fold + 1) + "_outof_5.csv"), sep="\t",
            header=0)
        X_train_video = prepare_video_modality_biLSTMAtt(X_train_video)
        X_test_video = prepare_video_modality_biLSTMAtt(X_test_video)

        X_train_video = X_train_video.sort_values(by=["name"])
        X_train_video = X_train_video.reset_index(drop=True)

        X_test_video = X_test_video.sort_values(by=["name"])
        X_test_video = X_test_video.reset_index(drop=True)

        # VIDEO AVG - STATIC MODELS
        X_train_video_avg = pd.read_csv(os.path.join(args.embs_dir_MLP, "train_fold"+str(fold)+".csv"), sep=";", header=0)
        X_test_video_avg = pd.read_csv(os.path.join(args.embs_dir_MLP, "test_fold" + str(fold) + ".csv"), sep=";",
                                        header=0)
        X_train_video_avg = X_train_video_avg.sort_values(by=["video_name"])
        X_train_video_avg = X_train_video_avg.reset_index(drop=True)

        X_test_video_avg = X_test_video_avg.sort_values(by=["video_name"])
        X_test_video_avg = X_test_video_avg.reset_index(drop=True)

        # AUDIO AVG - STATIC MODELS
        X_train_audio_avg = pd.read_csv(os.path.join(args.embs_dir_MLP, "train_fold" + str(fold) + ".csv"), sep=";",
                                        header=0)
        X_test_audio_avg = pd.read_csv(os.path.join(args.embs_dir_MLP, "test_fold" + str(fold) + ".csv"), sep=";",
                                       header=0)
        X_train_audio_avg = X_train_audio_avg.sort_values(by=["video_name"])
        X_train_audio_avg = X_train_audio_avg.reset_index(drop=True)

        X_test_audio_avg = X_test_audio_avg.sort_values(by=["video_name"])
        X_test_audio_avg = X_test_audio_avg.reset_index(drop=True)


        #Remove audio cols:
        y_train = pd.DataFrame([])
        y_test = pd.DataFrame([])
        y_train["emotion"] = X_train_video["emotion"]
        y_test["emotion"] = X_test_video["emotion"]
        #Remove cols:
        X_train_audio = X_train_audio.rename(columns={"name":"NAME", "emotion":"EMOTION"}) #"actor":"ACTOR"
        X_test_audio = X_test_audio.rename(columns={"name": "NAME", "emotion": "EMOTION"})  # "actor":"ACTOR"
        # print('audio_WAV2VEC',X_train_audio)
        # print('video_',X_train_video)
        # print('video_',X_train_video_avg)
        #Combine data
        #
        X_train_MM = pd.concat([X_train_audio, X_train_video,X_train_video_avg[["embs"+str(i) for i in range(8)]]], axis=1) #X_train_audio_avg[["embs"+str(i) for i in range(8)]]
        #
        X_test_MM = pd.concat([X_test_audio, X_test_video, X_test_video_avg[["embs"+str(i) for i in range(8)]]], axis=1) #X_test_audio_avg[["embs"+str(i) for i in range(8)]]

        #Remove columns
        X_train_MM = remove_cols(X_train_MM, cols2rm=["NAME", "EMOTION", "ACTOR",
                                                      "name", "emotion", "actor", "fold", "weigths"])

        X_test_MM = remove_cols(X_test_MM, cols2rm=["NAME", "EMOTION", "ACTOR",
                                                      "name", "emotion", "actor", "fold", "weigths"])

        # randomize data
        X_train_MM = shuffle(X_train_MM,random_state=seed)
        y_train = shuffle(y_train, random_state=seed)
        X_train_MM = X_train_MM.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)

        if (args.type_of_norm in [0, 1]):
            if (args.type_of_norm == 1):
                scaler = MinMaxScaler(feature_range=(0, 1))
            else:
                scaler = StandardScaler()
            X_train_MM = scaler.fit_transform(X_train_MM)
            X_test_MM = scaler.transform(X_test_MM)


        # print('X_train_MM', X_train_MM.shape, type(X_train_MM), X_train_MM[0])
        # print(y_train)
        # 设计训练集和验证集


        train_set = RAVDESSDataset(X_train_MM,y_train)
        test_set = RAVDESSDataset(X_test_MM,y_test)
        train_loader = DataLoader(dataset=train_set,batch_size=16,shuffle=True,num_workers=2)
        test_loader = DataLoader(dataset=test_set, batch_size=16, shuffle=True, num_workers=2)
        # print(train_set[0])
        model = MultiModalCNN(num_classes=8)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if
                                   p.requires_grad)

        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),  # 传入模型的参数
            lr=0.0004,  # 初始权重
            momentum=0.9,  # 设置动量参数，梯度采用累加的算法：震荡的方向的梯度互相抵消，梯度小的方向逐渐累加
            dampening=0.9,
            weight_decay=1e-3,  # 权重衰减
            nesterov=False)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(200):
            top1 = AverageMeter()
            losses = AverageMeter()
            top1_test = AverageMeter()
            losses_test = AverageMeter()
            for i,data in enumerate(train_loader,0):
                feature,labels = data
                audio = feature[:,0:8]
                video = feature[:,8:]
                # audio = torch.FloatTensor(audio)
                # video = torch.FloatTensor(video)
                m_pred,a_pred,v_pred = model(audio,video)
                # print(m_pred.shape,a_pred,v_pred,labels)
                labels = labels.view(-1)
                loss = criterion(m_pred,labels)+criterion(a_pred,labels)+criterion(v_pred,labels)
                losses.update(loss.data, feature.size(0))
                prec1, = calculate_accuracy(m_pred,labels,topk=(1,))
                top1.update(prec1, feature.size(0))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('train_epoch:', epoch, losses.avg.item(), top1.avg.item())
            wandb.log({'train_epoch': epoch,
                       'losses': losses.avg.item(),
                       'top1': top1.avg.item()})
            for i, data in enumerate(test_loader, 0):
                with torch.no_grad():
                    feature, labels = data
                    audio = feature[:, 0:8]
                    video = feature[:, 8:]
                m_pred, a_pred, v_pred = model(audio, video)
                # print(m_pred.shape,a_pred,v_pred,labels)
                labels = labels.view(-1)
                loss = criterion(m_pred, labels) + criterion(a_pred, labels) + criterion(v_pred, labels)
                losses_test.update(loss.data, feature.size(0))
                prec1, = calculate_accuracy(m_pred, labels, topk=(1,))
                top1_test.update(prec1, feature.size(0))
            print('test_epoch:', epoch, losses_test.avg.item(), top1_test.avg.item())
            wandb.log({'test_epoch':epoch,
                       'losses_test':losses_test.avg.item(),
                      'top1_test':top1_test.avg.item()})

        print("Total number of trainable parameters: ", pytorch_total_params)


        # Train models
        # classifier = get_classifier(args.model_number, args.param, seed=seed)
        # classifier.fit(X_train_MM, y_train)
        # print('训练网络')
        # print(classifier)

    #     if (get_embs):
    #         train_path = os.path.join(args.out_dir, "train_fold" + str(fold) + ".csv")
    #         extract_posteriors(classifier, X_train_MM, X_test_video_avg, train_path)
    #         test_path = os.path.join(args.out_dir, "test_fold" + str(fold) + ".csv")
    #         extract_posteriors(classifier, X_test_MM, X_test_video_avg, test_path)
    #
    #     predictions = classifier.predict(X_test_MM)
    #     accuracy = np.mean((y_test["emotion"] == predictions).astype(np.float)) * 100.
    #     avg_acc += accuracy
    #     print(f"Accuracy = {accuracy:.3f}")
    #     print("------------")
    # print("FINAL TEST ACCURACY: ", str(avg_acc / 5))
    #

    wandb.finish()

