# AttA-NET
ATTENTION AGGREGATION NETWORK FOR AUDIO-VISUAL EMOTION RECOGNITION

We provide implementations for RAVDESS dataset of speech and frontal face view data corresponding to 8 emotions: 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised. The way to use the project's code is as follows:

## Usage
## Dataset preparation
For training on Ravdess, download data from **https://zenodo.org/record/1188976#.YkgJVijP2bh**. You will need to download the files Video_Speech_Actor_[01-24].zip and Audio_Speech_Actors_01-24.zip. The directory should be organized as follows:
    RAVDESS
    └───ACTOR01
    │   │  01-01-01-01-01-01-01.mp4
    │   │  01-01-01-01-01-02-01.mp4
    │   │  ...
    │   │  03-01-01-01-01-01-01.wav
    │   │  03-01-01-01-01-02-01.wav
    │   │  ...
    └───ACTOR02
    └───...
    └───ACTOR24

Install face detection library:

    pip install facenet-pytorch

or follow instructions in https://github.com/timesler/facenet-pytorch

Preprocessing scripts are located in ravdess_preprocessing/ Inside each of three scripts, specify the path (full path!) where you have downloaded the data. Then run:
    
    cd ravdess_preprocessing
    python extract_faces.py
    python extract_audios.py
    python create_annotations.py



