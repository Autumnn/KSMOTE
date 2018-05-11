from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import Read_Data as RD

#dir = "wine-5-fold/wine-5-1tra.dat"
dir = "KSMOTE_IECON15_InputData.csv"

RD.Initialize_Data(dir)

for i in range(0, 9):
    for j in range(0, 9):
        if i != j:
            fig = plt.figure()
            p1 = plt.hist2d(RD.Features[:,i], RD.Features[:,j], bins = 30)

            File_name = "2D_Distributions_of_" + str(i) + "_and_" + str(j) + "_Feature.png"
            fig.savefig(File_name)