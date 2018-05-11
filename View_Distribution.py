from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import Read_Data as RD

#dir = "wine-5-fold/wine-5-1tra.dat"
dir = "KSMOTE_IECON15_InputData.csv"

RD.Initialize_Data(dir)


for j in range(RD.Num_Features):
    MIN = min(RD.Features[:, j])
    MAX = max(RD.Features[:, j])
    Bin = np.linspace(MIN, MAX, 100)
    fig = plt.figure()
    Title = "Histogram of the " + str(j) + "th Feature"
    fig.canvas.set_window_title(Title)
    fig.subplots_adjust(hspace=0.4)

    ax = plt.subplot(2,1,1)
    ax.set_title("Positive")
    ax.hist(RD.Positive_Feature[:,j], bins = Bin, facecolor='yellowgreen')

    ax = plt.subplot(2,1,2)
    ax.set_title("Negative")
    ax.hist(RD.Negative_Feature[:,j], bins = Bin, facecolor='blue')

    File_name = "Histogram_of _the_" + str(j) + "th_Feature.png"
    fig.savefig(File_name)

