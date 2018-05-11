from __future__ import print_function
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors as mcolors
import numpy as np
import Read_Data as RD

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

#dir = "wine-5-fold/wine-5-1tra.dat"
dir = "KSMOTE_IECON15_InputData.csv"

RD.Initialize_Data(dir)

Features_Attribute = [5, 6, 7, 8]
#Features_Attribute = np.arange(0, RD.Num_Features, 1)

l = len(Features_Attribute)

for i in range(0, l):
    for j in range(i+1, l):
        for k in range(j+1, l):
            X_index = Features_Attribute[i]
            Y_index = Features_Attribute[j]
            Z_index = Features_Attribute[k]
            print(X_index, Y_index, Z_index)
            ax = plt.subplot(111, projection='3d')
            ax.scatter(RD.Stage_1_Feature[:,X_index], RD.Stage_1_Feature[:,Y_index], RD.Stage_1_Feature[:,Z_index], marker = 'o', color = '#539caf', label='1', s = 10, alpha=0.4)
            ax.scatter(RD.Stage_2_Feature[:, X_index], RD.Stage_2_Feature[:, Y_index], RD.Stage_2_Feature[:, Z_index], marker = '+', color = colors["forestgreen"], label='2', s = 20, alpha=0.6)
            ax.scatter(RD.Stage_3_Feature[:, X_index], RD.Stage_3_Feature[:, Y_index], RD.Stage_3_Feature[:, Z_index], marker = 'd', color = colors["darkmagenta"], label='3', s = 30, alpha=0.8)
            ax.scatter(RD.Stage_4_Feature[:, X_index], RD.Stage_4_Feature[:, Y_index], RD.Stage_4_Feature[:, Z_index], marker = '^', color = 'r', label='4', s = 40, alpha=0.9)
            plt.show()