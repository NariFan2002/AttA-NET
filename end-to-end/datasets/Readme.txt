1.下载数据
关于Ravdess的培训，从这里下载数据。https://zenodo.org/record/1188976#.YkgJVijP2bh
你需要下载文件Video_Speech_Actor_[01-24].zip和Audio_Speech_Actors_01-24.zip。该目录应按如下方式组织:
第一个数：01代表既有音频也有视频.mp4，02是只有视频.m[4，03代表只有音频结尾是.wav

RAVDESS
└───ACTOR01
│   │  01-01-01-01-01-01-01.mp4
│   │  01-01-01-01-01-02-01.mp4
│   │  ...
│   │  02-01-01-01-01-01-01.mp4
│   │  02-01-01-01-01-02-01.mp4
│   │  ...
└───ACTOR02
└───...
└───ACTOR24
└───Audio_Speech_Actors_01-24
└──────ACTOR01
    │   │  03-01-01-01-01-01-01.wav
    │   │  03-01-01-01-01-02-01.wav
    │   │  ...

## 我的数据集放在：/home/fanrj/Datasets/RAVDESS

2.安装人脸检测库
pip install facenet-pytorch
来自论文：https://github.com/timesler/facenet-pytorch

3.查看RAVDESS数据集的处理方式
处理方式写在 multimodal-emotion-recognition-AV-RAVDESS/ravdess_preprocessing/ 文件夹中
修改数据集存放的路径后
运行：
cd ravdess_preprocessing
python extract_faces.py
python extract_audios.py
python create_annotations.py