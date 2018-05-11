from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np

def Initialize_Data(dir):
    Num_lines = len(open(dir, 'r').readlines())
    num_columns = 0
    data_info_lines = 0
    with open(dir, "r") as get_info:
        print("name", get_info.name)
        for line in get_info:
            if line.find("\"RFHRS\"") == 0:
                data_info_lines += 1
            else:
                columns = line.split(",")
                num_columns = len(columns)
                break

    global Num_Samples
    Num_Samples = Num_lines - data_info_lines
    print(Num_Samples)
    global Num_Features
    Num_Features = num_columns - 1

    global Features
    Features = np.ones((Num_Samples, Num_Features))
    global Labels
    Labels = np.ones((Num_Samples, 1))

    global Num_stage_1
    Num_stage_1 = 0
    global Num_stage_2
    Num_stage_2 = 0
    global Num_stage_3
    Num_stage_3 = 0
    global Num_stage_4
    Num_stage_4 = 0

    with open(dir, "r") as data_file:
        print("Read Data", data_file.name)
        l = 0
        for line in data_file:
            l += 1
            if l > data_info_lines:
                # print(line)
                row = line.split(",")
                length_row = len(row)
                # print('Row length',length_row)
                # print(row[0])
                for i in range(length_row):
                    if i < length_row - 1:
                        Features[l - data_info_lines - 1][i] = row[i]
                        # print(Features[l-14][i])
                    else:
                        attri = row[i].strip()
                        # print(attri)
                        # if attri == '\"Stage 1\"' or attri == '\"Stage 2\"':        # A
                        if attri == '\"Stage 1\"':  # B
                            Labels[l - data_info_lines - 1][0] = 1
                            Num_stage_1 += 1
                        elif attri == '\"Stage 2\"':
                            Labels[l - data_info_lines - 1][0] = 2
                            Num_stage_2 += 1
                        elif attri == '\"Stage 3\"':
                            Labels[l - data_info_lines - 1][0] = 3
                            Num_stage_3 += 1
                        else:
                            Labels[l - data_info_lines - 1][0] = 4
                            Num_stage_4 += 1

    global Stage_1_Feature
    Stage_1_Feature = np.ones((Num_stage_1, Num_Features))
    global Stage_2_Feature
    Stage_2_Feature = np.ones((Num_stage_2, Num_Features))
    global Stage_3_Feature
    Stage_3_Feature = np.ones((Num_stage_3, Num_Features))
    global Stage_4_Feature
    Stage_4_Feature = np.ones((Num_stage_4, Num_Features))

    index_s_1 = 0
    index_s_2 = 0
    index_s_3 = 0
    index_s_4 = 0

    for i in range(Num_Samples):
        if Labels[i] == 1:
            Stage_1_Feature[index_s_1] = Features[i]
            index_s_1 += 1
        elif Labels[i] == 2:
            Stage_2_Feature[index_s_2] = Features[i]
            index_s_2 += 1
        elif Labels[i] == 3:
            Stage_3_Feature[index_s_3] = Features[i]
            index_s_3 += 1
        else:
            Stage_4_Feature[index_s_4] = Features[i]
            index_s_4 += 1

#    print("Number of Positive: ", Num_positive)
    global Positive_Feature
    Positive_Feature = np.concatenate((Stage_3_Feature, Stage_4_Feature))
#    print("Num of Negative: ", Num_negative)

    print("Read Completed")

def get_feature():
    return Features

def get_label():
    return  Labels

def get_positive_feature():
    return Positive_Feature


