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

As a result you will have annotations.txt file that you can use further for training.

## Training
For the initialization of weights in the visual module, please download the pre-trained EfficientFace model from the provided link under 'Pre-trained models.' In our experiments, we utilized the model that was pre-trained on AffectNet7, specifically the 'EfficientFace_Trained_on_AffectNet7.pth.tar' model. Alternatively, you can choose to skip this step and train the model from scratch, although it is likely to result in lower performance.

For training，run：
        python main.py

You can optionally specify the arguments in end-to-end/opts.py. By default, this will train the model, select the best iteration based on the validation set, and provide a performance report for the test set. Any other training parameters that require adjustment should be self-explanatory.

## Testing
If you wish to test a previously trained model, please specify the '--no_train' and '--no_val' arguments, along with the path to the experiment folder containing the checkpoint:
        python main.py  --no_train --no_val --result_path [PATH_TO_CHECKPOINT_FOLDER]




