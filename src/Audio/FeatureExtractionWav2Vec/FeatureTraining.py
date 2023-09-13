"""
    Script to train the models (MLP/SVC...) with the average embedding (512-dimensional) calculated from the baseline model (jonatasgrosman/wav2vec2-large-xlsr-53-english)
	author: Cristina Luna.
	date: 03/2022

	Usage:
		e.g.
		 python3 MMEmotionRecognition/src/Audio/FeatureExtractionWav2Vec/FeatureTraining.py
		 --embs_dir <RAVDESS_dir>/FineTuningWav2Vec2_embs512
		 --model_number 1
		 --param 0.01
		 --type_of_norm 0
		 --out_dir <RAVDESS_dir>/posteriors_embs512_SVC_C001
	Options:
         --embs_dir Path with the embeddings to train/test the models
		 --model_number: Number to identify the model to train and test [1-11]: 1-SVC / 2- Logistic Regression / 3- ridgeClassifier /4-perceptron / 5-NuSVC / 6-LinearSVC / 7-knn / 8-NearestCentroid / 9- DecrissionTree / 10- RandomForest / 11 - MLP')
		 --param: Parameter of the model: C in SVC / C in Logistic Regression / alpha in ridgeClassifier / alpha in perceptron / nu in NuSVC / C in LinearSVC / k in knn / None in NearestCentroid / min_samples_split in DecrissionTree / n_estimators in RandomForest / hidden_layer_sizes in MLP
		 --type_of_norm: Normalizaton to apply: '0-MinMax Norm / 1-Standard Norm / 2- No apply normalization [default: 2]
		 --out_dir : Path to save the posteriors of the trained models
"""


import os
import sys
import argparse
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.Audio.FineTuningWav2Vec.main_FineTuneWav2Vec_CV import generate_train_test



def clean_df(df):
    """
    Process dataframe to split features and labels to train the model. It returns the features, the labels and the name of the
    videos in 3 different dataframes.

    :param df:[DataFrame] Dataframe with the complete list of files generated by prepare_RAVDESS_DS(..) function
    """
    df["emotion"] = pd.to_numeric(df["emotion"])
    video_name_col = df["video_name"]
    # remove columns:
    df = df.drop(["index", "path", "video_name", "actor"], axis=1)
    labels = df["emotion"]
    features = df.drop(["emotion"], axis=1)
    return features, labels, video_name_col




def get_classifier(model, param, seed = 2020):
    """
    Process dataframe to split features and labels to train the model. It returns the features, the labels and the name of the
    videos in 3 different dataframes.

    :param model:[int] Number of the model to use. 1-SVC / 2- Logistic Regression / 3- ridgeClassifier /4-perceptron
    / 5-NuSVC / 6-LinearSVC / 7-knn / 8-NearestCentroid / 9- DecrissionTree / 10- RandomForest / 11 - MLP')
    :param param:[str] Parameter of the model: C in SVC / C in Logistic Regression / alpha in ridgeClassifier / alpha in perceptron
    / nu in NuSVC / C in LinearSVC / k in knn / None in NearestCentroid / min_samples_split in DecrissionTree / n_estimators in RandomForest / hidden_layer_sizes in MLP
    :param seed:[int] Seed to initialize the random seed generators
    """
    if model == 1:
        print("SVC ")
        classifier = SVC(random_state=seed, C=float(param))
    elif model == 2:
        print("LOGISTIC REGRES. ")
        classifier = LogisticRegression(random_state=seed, max_iter=10000, C=float(param))
    elif model == 3:
        print("RIDGE CLASSIF. ")
        classifier = RidgeClassifier(random_state=seed, alpha=float(param))
    elif model == 4:
        classifier = Perceptron(random_state=seed, alpha=float(param))
    elif model == 5:
        print("NU SVC ")
        classifier = NuSVC(random_state=seed, nu=float(param))
    elif model == 6:
        print("LINEAR SVC")
        classifier = LinearSVC(random_state=seed, max_iter=10000, C=float(param))
    elif model == 7:
        print("KNN")
        classifier = KNeighborsClassifier(n_neighbors=int(param))
    elif model == 8:
        print("NEAREST CENTROID")
        classifier = NearestCentroid()
    elif model == 9:
        print("DECISSION TREE")
        classifier = sklearn.tree.DecisionTreeClassifier(random_state=seed, min_samples_split=int(param))
    elif model == 10:
        print("RANFOM FOREST")
        classifier = RandomForestClassifier(random_state=seed, n_estimators=int(param))
    elif model == 11:
        print("MLP")
        classifier = MLPClassifier(random_state=seed, hidden_layer_sizes=eval(param),learning_rate='adaptive',learning_rate_init=0.00004,max_iter=700,warm_start=True,verbose=True)  # learning_rate_init=0.05

    else:
        print('error')
    return classifier



def extract_posteriors(model, X_data, df_names, out_path):
    """
    Extract posteriors from the trained models.

    :param model:[sklearn model] Trained model of the sklearn library
    :param X_data:[DataFrame] Dataframe with the embeddings to generate the posteriors
    :param df_names:[DataFrame] Dataframe with the names of the videos associated to the embeddings of X_data
    :param out_path:[str] Path to save the posteriors
    """
    out_prob = model.predict_proba(X_data)
    df_aux = pd.DataFrame(out_prob, columns=["embs"+str(i) for i in range(8)])
    df_aux["video_name"] = df_names["video_name"]
    df_aux.to_csv(out_path, sep=";", header=True, index=False)


def process_Wav2Vec512embs(embs_path, avg_embs_path):
    """
    Process dataframe to split features and labels to train the model. It returns the features, the labels and the name of the
    videos in 3 different dataframes.

    :param embs_path:[str] Path to read the embeddings extracted from Wav2Vec2.0. This embeddings has dimension (512,timesteps)
    :param avg_embs_path:[str] Path to save the average embedding calculated collapsing the timesteps. This embeddings has dimension (512,1)
    """
    X_total = pd.DataFrame([])
    for video_embs in os.listdir(embs_path):
        embs_df = pd.read_csv(os.path.join(embs_path, video_embs), sep=";", header=0)
        #Calculate average
        aux_df = pd.DataFrame([embs_df.mean()], columns=embs_df.columns)
        #Add other information
        aux_df["video_name"] = video_embs.split(".")[0]
        aux_df["path"] = video_embs.split(".")[0]
        aux_df["index"] = 0
        aux_df["actor"] = int(video_embs.split("-")[-1].split(".")[0])
        aux_df["emotion"] = int(video_embs.split("-")[2])-1
        X_total = X_total.append(aux_df)
    X_total.to_csv(avg_embs_path, sep=";", header=True, index=False)
    return X_total



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-embs', '--embs_dir', type=str, required=True,
                        help='Path with the embeddings to train/test the models')
    parser.add_argument('-out', '--out_dir', type=str, help='Path to save the trained model and the posteriors extracted from the model [default: Not save embs]',
                        default='')
    parser.add_argument('-m', '--model_number', type=int, required=True,
                        help='1-SVC / 2- Logistic Regression / 3- ridgeClassifier /4-perceptron / 5-NuSVC / 6-LinearSVC / 7-knn / 8-NearestCentroid / 9- DecrissionTree / 10- RandomForest / 11 - MLP')
    parser.add_argument('-modelParam', '--param', type=str, required=True,
                        help='Parameter of the model: C for SVC / C Logistic Regression / alpha in ridgeClassifier / alpha in perceptron / nu in NuSVC / C in LinearSVC / k in knn / None in NearestCentroid / min_samples_split in DecrissionTree / n_estimators in RandomForest / hidden_layer_sizes in MLP', default=2)
    parser.add_argument('-norm', '--type_of_norm', type=int, required=True,
                        help='0-MinMax Norm / 1-Standard Norm / 2- No apply normalization [default: 2]', default=2)


    args = parser.parse_args()
    #param = (80)  # Parameter of each type of model (Check get_classifier(...) function)
    seed = 2023

    avg_embs_path = os.path.join(args.embs_dir.rsplit("/", 1)[0], "avg_embs_512.csv")
    if (eval(args.out_dir) == ""):
        get_embs = False
    else:
        get_embs = True
        os.makedirs(args.out_dir, exist_ok=True)
    # Get average embedding per video:
    if(os.path.exists(avg_embs_path)):
        X_total = pd.read_csv(avg_embs_path, sep=";", header=0)
    else:
        X_total = process_Wav2Vec512embs(args.embs_dir, avg_embs_path)


    #Start 5-CV strategy
    avg_acc = 0
    for fold in range(1):
        print("Processing fold: ", str(fold))
        #Generate train/test sets 生成训练/测试集
        train_df, test_df = generate_train_test(fold, X_total)
        X_train, y_train, _ = clean_df(train_df)
        X_test, y_test, _ = clean_df(test_df)

        #Normalize (if required) 规范化(如果需要)
        if (int(args.type_of_norm) in [0, 1]):
            if (args.type_of_norm == 1):
                scaler = MinMaxScaler(feature_range=(0, 1))
            else:
                scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Create and train classifier 创建和训练分类器
        classifier = get_classifier(args.model_number, args.param, seed = 2020)
        classifier.fit(X_train, y_train)

        #Extract and save posteriors of the trained models of the train and test sets
        if (get_embs):
            train_path = os.path.join(args.out_dir, "train_fold" + str(fold) + ".csv")
            extract_posteriors(classifier, X_train, train_df, train_path)
            test_path = os.path.join(args.out_dir, "test_fold" + str(fold) + ".csv")
            extract_posteriors(classifier, X_test, test_df, test_path)

        # Evaluate using the trained classifier
        predictions = classifier.predict(X_test)
        accuracy = np.mean((y_test == predictions).astype(np.float)) * 100.
        avg_acc+=accuracy
        print(f"Accuracy = {accuracy:.3f}")
        print("------------")
    print("FINAL TEST ACCURACY: ", str(avg_acc/5))
