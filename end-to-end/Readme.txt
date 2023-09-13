第一步：安装依赖配置环境：
pip install -r requirements.txt
# CUDA 10.2，linux or Windows 情况下配置 torch=1.10.0
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu102/torch_stable.html
其他系统配置见：https://pytorch.org/get-started/previous-versions/

第二步：
进入ravdess_preprocess文件夹,对数据进行预处理（就是每个视频提取15张图片，每个音频剪辑成同样的长度）

第三步：
dataset.py文件用于读取ravdess dataset （底层工具在 datasets.ravdess）
model.py文件用于建立模型 并且返回函数和函数的参数 函数：generate_model （底层工具在models.multimodalcnn）
opts.py文件用于设置一些命令行的配置
utils.py文件给出了一些工具函数