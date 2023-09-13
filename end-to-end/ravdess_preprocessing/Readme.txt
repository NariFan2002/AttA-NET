这个ravdess_preprocessing文件夹中是对该数据集进行预处理的操作；
每个视频提取15张人脸图片
每个视频提取3.6s的音频内容

RAVDESS的这一部分包含1440个文件:每个演员60个试验x 24个演员= 1440。
《RAVDESS》包含24名专业演员(12名女性，12名男性)，用中性的北美口音发音两个词汇匹配的语句。
言语情感包括平静、快乐、悲伤、愤怒、恐惧、惊讶和厌恶的表情。
每一种表情都有两种情绪强度(正常、强烈)，外加一种中性的表情。

标注方法：
Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
Vocal channel (01 = speech, 02 = song).
Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
Repetition (01 = 1st repetition, 02 = 2nd repetition).
Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
Filename example: 03-01-06-01-02-01-12.wav

Audio-only (03)
Speech (01)
Fearful (06)
Normal intensity (01)
Statement "dogs" (02)
1st Repetition (01)
12th Actor (12)
Female, as the actor ID number is even.


运行：
cd ravdess_preprocessing
python extract_faces.py
python extract_audios.py
python create_annotations.py

运行之后会得到annotations.txt里面包含了处理好的人脸数据(.npy)、定长的音频数据(.wav)和标签值（1-8）