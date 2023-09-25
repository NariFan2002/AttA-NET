Step 1: Install Dependencies and Configure Environment:
    
    pip install -r requirements.txt

# For CUDA 10.2, on Linux or Windows, configure torch=1.10.0 as follows:

    pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu102/torch_stable.html

For configurations on other systems, refer to: https://pytorch.org/get-started/previous-versions/

Step 2:
Go into the 'ravdess_preprocess' folder and preprocess the data (i.e., extract 15 images from each video and clip each audio to the same length).

Step 3:
The 'dataset.py' file is used to read the RAVDESS dataset (underlying tools are in 'datasets.ravdess').
The 'model.py' file is used to build the model and return functions and their parameters. Function: 'generate_model' (underlying tools are in 'models.multimodalcnn').
The 'opts.py' file is used to configure some command-line settings.
The 'utils.py' file provides various utility functions.

