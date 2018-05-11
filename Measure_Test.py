import os
import numpy as np
import xgboost as xgb
from sklearn import svm, preprocessing, metrics
from sklearn.multiclass import OneVsRestClassifier
#from keras.models import load_model
#from imblearn.over_sampling import SMOTE
#import cGANStructure
#from RACOG import RACOG
from metrics_list import metric_list

path = "KSMOTE_3_M_Cross_Folder_npz"
dirs = os.listdir(path) #Get files in the folder

for Dir in dirs:
    print("Data Set Name: ", Dir)
    dir_path = path + "/" + Dir
    files = os.listdir(dir_path)  # Get files in the folder
    f_i = 0
    par_a = []
    par_b = []


    Num_Gamma = 100
    gamma = np.logspace(-4, 2, Num_Gamma)
    Num_C = 100
    C = np.logspace(-2, 4, Num_C)
    Num_Cross_Folders = 3
    ml_record = metric_list(gamma, C, Num_Cross_Folders)


    '''
    max_depth = np.arange(3,10,1)           # best = 3
    min_child_weight = np.arange(1,6,1)     # best = 1
    gamma = np.arange(0,1,0.1)              # best = 0.0
    subsample = np.arange(0.5,1,0.1)        # best = 0.5
    colsample_bytree = np.arange(0.5,1,0.1) # best = 0.6
    reg_alpha = np.logspace(-5, 5, 10)      # best = 0.002
    learning_rate = np.logspace(-2, 0, 10)  # best = 0.215
    '''

    Num_Cross_Folders = 3
    ml_record = metric_list(gamma, C, Num_Cross_Folders)


    i = 0
    for file in files:
        name = dir_path + '/' + file
        r = np.load(name)

        Positive_Features_train = r["P_F_tr"]
        Num_Positive_train = Positive_Features_train.shape[0]
        Positive_Labels_train = np.linspace(1, 1, Num_Positive_train)

        Positive_Features_test = r["P_F_te"]
        Num_Positive_test = Positive_Features_test.shape[0]
        Positive_Labels_test = np.linspace(1, 1, Num_Positive_test)

        Stage_1_Features_train = r["N_1_tr"]
        Num_S_1_train = Stage_1_Features_train.shape[0]
        Stage_1_Labels_train = np.linspace(0, 0, Num_S_1_train)

        Stage_1_Features_test = r["N_1_te"]
        Num_S_1_test = Stage_1_Features_test.shape[0]
        Stage_1_Labels_test = np.linspace(0, 0, Num_S_1_test)

        Stage_2_Features_train = r["N_2_tr"]
        Num_S_2_train = Stage_2_Features_train.shape[0]
        Stage_2_Labels_train = np.linspace(-1, -1, Num_S_2_train)

        Stage_2_Features_test = r["N_2_te"]
        Num_S_2_test = Stage_2_Features_test.shape[0]
        Stage_2_Labels_test = np.linspace(-1, -1, Num_S_2_test)

        print(i, " folder; ", "Po_tr: ", Num_Positive_train, "N_1_tr: ", Num_S_1_train, "N_2_tr: ", Num_S_2_train,
              "Po_te: ", Num_Positive_test, "N_1_te: ", Num_S_1_test, "N_2_te: ", Num_S_2_test)

        Features_train_o = np.concatenate((Positive_Features_train, Stage_1_Features_train, Stage_2_Features_train))
        Labels_train_o = np.concatenate((Positive_Labels_train, Stage_1_Labels_train, Stage_2_Labels_train))
        #                print(Labels_train_o)
        Feature_test = np.concatenate((Positive_Features_test, Stage_1_Features_test, Stage_2_Features_test))
        Label_test = np.concatenate((Positive_Labels_test, Stage_1_Labels_test, Stage_2_Labels_test))

#        for j in range(len(gamma)):
#            for k in range(len(C)):
                # print("gamma = ", str(gamma[j]), " C = ", str(C[k]))
#                clf = svm.SVC(C=C[k], kernel='rbf', gamma=gamma[j])
#        clf = svm.SVC(C=1.748, kernel='rbf', gamma=28.48)


        clf = xgb.XGBClassifier(learning_rate = 0.002,
                                n_estimators=10000,
                                max_depth= 3,
                                min_child_weight= 1,
                                gamma= 0,
                                subsample= 0.5,
                                colsample_bytree= 0.6,
                                objective= 'binary:logistic',
                                reg_alpha = 0.215,
                                scale_pos_weight=1,
                                seed=27)

#        clf.fit(Features_train_o, Labels_train_o)
        ova = OneVsRestClassifier(clf)
        ova.fit(Features_train_o, Labels_train_o)
#                Label_predict = clf.predict(Feature_test)
        Label_predict = ova.predict(Feature_test)
        print(Label_test)
        print(Label_predict)
#                ml_record.measure(j, k, i, Label_test, Label_predict)
#                Label_score = clf.predict_proba(Feature_test)
#                Label_score = ova.predict_proba(Feature_test)
#                Label_score = ova.decision_function(Feature_test)
#                ml_record.auc_measure(j, k, i, Label_test, Label_score[:,1])

    i += 1
#    file_wirte = "Result_SVM_KSMOTE.txt"
#    ml_record.output(file_wirte, "none", Dir)