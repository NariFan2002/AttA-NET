# coding='utf-8'
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

def draw_tsne(feature, target, Prediction_map):
    # feature 是特征
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(feature)
    data = result
    label = target

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    plt.xlim(0, 1.05)  # 固定 x 轴范围在 2 到 8
    plt.ylim(0, 1.05)  # 固定 y 轴范围在 -1 到 1
    # ax1 = plt.subplot(1, 1, 1)
    for i in range(data.shape[0]):
        # if Prediction_map[i] != target[i]:
        x, y = data[i, 0], data[i, 1]
        fontdict = {'weight': 'bold', 'size': 9}
        # bbox = fig.get_window_extent()
        num = int(label[i])
        color = plt.cm.Set1(num)
        # text_width = len(str(num)) * fontdict['size'] * 0.6
        # text_height = fontdict['size']  # 假设文本高度与字体大小相等
        # if x + text_width > bbox.width:
        #     x -=text_width
        # if y + text_height > bbox.height:
        #     y -= text_height  # 如果高度超出边框，向下偏移数字的位置
        plt.text(x, y, str(int(label[i])),color=color,fontdict=fontdict)
        # plt.plot(data[i, 0], data[i, 1], marker='o', mfc='none', color='red', markersize=9)
        '''
        else:
            plt.text(data[i, 0], data[i, 1], str(int(label[i])),
                     color=plt.cm.Set1(int(label[i])),
                     fontdict={'weight': 'bold', 'size': 9})
        '''

    plt.xticks([])  # ignore xticks
    plt.yticks([])  # ignore yticks
    return fig