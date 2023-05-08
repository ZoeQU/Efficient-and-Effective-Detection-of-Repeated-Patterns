# -*- coding:utf-8 -*-
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import maximum_filter, gaussian_filter
from scipy.stats import multivariate_normal
from skimage.feature import peak_local_max


def visualizeV(V, savename):
    if type(V) is np.ndarray:
        plt.imshow(V)
        plt.savefig(savename)
        # plt.show()
        plt.close()

    else:
        n = len(V)
        if n > 1:
            nn = math.ceil(n / 3.0)
            for ii in range(n):
                plt.subplot(3, nn, ii + 1)
                plt.imshow(V[ii])
            plt.savefig(savename)
            # plt.show()
            plt.close()
        else:
            plt.imshow(V)
            plt.savefig(savename)
            # plt.show()
            plt.close()


def sim_curve(resx, savename, title):
    """plot similarity curve"""
    x = list(range(resx.shape[1]))
    y = resx.tolist()[0]
    plt.figure(figsize=(10, 8))
    plt.plot(x, y, color="red", linewidth=1, linestyle="--")
    plt.xlabel("pixel")  # xlabel、ylabel：分别设置X、Y轴的标题文字。
    plt.ylabel("sim")
    plt.title(title)  # title：设置子图的标题。
    plt.savefig(savename, dpi=120, bbox_inches='tight')
    # plt.show()
    plt.close()