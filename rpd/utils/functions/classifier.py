# -*- coding:utf-8 -*-
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from sklearn.cluster import MeanShift, estimate_bandwidth
from matplotlib import pyplot as plt

# TODO(zoe) not finished now

def classifier(img):
    # # # cluster color
    wa = img.shape[1]
    ha = img.shape[0]
    # # 1. x-axis
    temp_x = img[0:10, :wa]
    colorsx, countsx = np.unique(temp_x.reshape(-1, 3), axis=0, return_counts=1)
    bandwidthx = estimate_bandwidth(colorsx, quantile=0.3)
    # print(bandwidthx)
    msx = MeanShift(bandwidth=bandwidthx, bin_seeding=True)
    msx.fit(colorsx)
    labelsx = msx.labels_
    cluster_colorsx = np.rint(msx.cluster_centers_)
    n_clusters_x = len(cluster_colorsx)

    # # 2. y-axis
    temp_y = img[0:ha, 0:10]
    colorsy, countsy = np.unique(temp_y.reshape(-1, 3), axis=0, return_counts=1)
    bandwidthy = estimate_bandwidth(colorsy, quantile=0.3)
    # print(bandwidthy)
    msy = MeanShift(bandwidth=bandwidthy, bin_seeding=True)
    msy.fit(colorsy)
    labels = msy.labels_
    cluster_colorsy = np.rint(msy.cluster_centers_)
    n_clusters_y = len(cluster_colorsy)

    # # 3. classify
    if n_clusters_x > 1 and n_clusters_y > 1:
        return 'quadrilateral'
    else:
        return 'two-dimension'
