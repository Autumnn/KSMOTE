from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import Read_Data as RD
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold


dir = "KSMOTE_IECON15_InputData.csv"
RD.Initialize_Data(dir)
name = dir.split(".")[0]

print(RD.Positive_Feature[0])
print(RD.Positive_Feature.shape)
print(RD.Stage_1_Feature[0])
print(RD.Stage_1_Feature.shape)
print(RD.Stage_2_Feature[0])
print(RD.Stage_2_Feature.shape)

npy_name = name + "_M.npz"
np.savez(npy_name, P_F = RD.Positive_Feature, N_1_F = RD.Stage_1_Feature, N_2_F = RD.Stage_2_Feature)
file = npy_name

print("File Name: ", file)
name = file.split(".")[0]
dir = file
r = np.load(dir)

Positive_Features = r["P_F"]
Num_Positive = Positive_Features.shape[0]
Positive_Labels = np.linspace(1,1,Num_Positive)
Stage_1_Features = r["N_1_F"]
Num_S_1 = Stage_1_Features.shape[0]
Stage_1_Labels = np.linspace(0,0,Num_S_1)
Stage_2_Features = r["N_2_F"]
Num_S_2 = Stage_2_Features.shape[0]
Stage_2_Labels = np.linspace(-1,-1,Num_S_2)

Features = np.concatenate((Positive_Features, Stage_1_Features, Stage_2_Features))
Labels = np.concatenate((Positive_Labels, Stage_1_Labels, Stage_2_Labels))

Num_Cross_Folders = 3
min_max_scalar = preprocessing.MinMaxScaler()
Re_Features = min_max_scalar.fit_transform(Features)

#skf = StratifiedKFold(n_splits=Num_Cross_Folders, shuffle=True)
skf = StratifiedKFold(n_splits=Num_Cross_Folders, shuffle=False)

i = 0
for train_index, test_index in skf.split(Re_Features, Labels):
    Feature_train, Feature_test = Re_Features[train_index], Re_Features[test_index]
    Label_train, Label_test = Labels[train_index], Labels[test_index]

    Positive_Feature_train = Feature_train[np.where(Label_train == 1)]
    Positive_Feature_test = Feature_test[np.where(Label_test == 1)]
    Stage_1_Features_train = Feature_train[np.where(Label_train == 0)]
    Stage_1_Features_test = Feature_test[np.where(Label_test == 0)]
    Stage_2_Features_train = Feature_train[np.where(Label_train == -1)]
    Stage_2_Features_test = Feature_test[np.where(Label_test == -1)]

    saved_name = name + "_" + str(i) + "_M_Cross_Folder.npz"
    np.savez(saved_name, P_F_tr = Positive_Feature_train, P_F_te = Positive_Feature_test,
             N_1_tr = Stage_1_Features_train, N_1_te = Stage_1_Features_test,
             N_2_tr = Stage_2_Features_train, N_2_te = Stage_2_Features_test)

    i += 1