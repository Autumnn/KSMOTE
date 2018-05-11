from __future__ import print_function
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import Read_Data as RD

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

#dir = "wine-5-fold/wine-5-1tra.dat"
dir = "KSMOTE_IECON15_InputData.csv"

RD.Initialize_Data(dir)

for i in range(0, RD.Num_Features):
    for j in range(i+1, RD.Num_Features):
        if i != j:
            fig = plt.figure()
            p1 = plt.scatter(RD.Stage_1_Feature[:,i], RD.Stage_1_Feature[:,j], marker = 'o', color = '#539caf', label='1', s = 10, alpha=0.4)
            p2 = plt.scatter(RD.Stage_2_Feature[:,i], RD.Stage_2_Feature[:,j], marker = '+', color = colors["forestgreen"], label='2', s = 20, alpha=0.6)
            p3 = plt.scatter(RD.Stage_3_Feature[:,i], RD.Stage_3_Feature[:,j], marker = 'd', color = colors["darkmagenta"], label='3', s = 30, alpha=0.8)
            p4 = plt.scatter(RD.Stage_4_Feature[:,i], RD.Stage_4_Feature[:,j], marker = '^', color = 'r', label='4', s = 40, alpha=0.9)
            File_name = "Scatter_Plot_of_" + str(i) + "_and_" + str(j) + "_Feature.png"
            fig.savefig(File_name)


