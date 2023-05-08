# -*- coding:utf-8 -*-
import numpy as np
import cv2
import random
from skimage.feature import peak_local_max


def rand_peaks(map):
    x = range(map.shape[0])
    y = range(map.shape[1])
    if map.shape[0] > 20 and map.shape[1] > 20:
        p_x = np.array(random.sample(x[1:], 10)).reshape(-1, 1)
        p_y = np.array(random.sample(y[1:], 10)).reshape(-1, 1)
        peaks_l = np.hstack((p_x, p_y))
    else:
        p_x = np.array(x[0: len(x): 2]).reshape(-1, 1)
        p_y = np.array(y[0: len(y): 2]).reshape(-1, 1)
        if len(p_x) != len(p_y):
            l = min(len(p_x), len(p_y))
            p_x = p_x[: l-1]
            p_y = p_y[: l-1]
        peaks_l = np.hstack((p_x, p_y))
    return peaks_l


def select_peaks_l(map, area_para, num):
    peaks_l = peak_local_max(map, area_para)
    if len(peaks_l) > num:
        return peaks_l
    else:
        area_para = area_para - 1
        if area_para > 0:
            peaks_l = select_peaks_l(map, area_para, num)
        else:
            peaks_l = rand_peaks(map)
            return peaks_l
    return peaks_l


def show_peaks(peaks, li, image_path, image_name, save_path):
    save_name = save_path + image_name + '_peaks_' + str(li) + '.png'
    im = cv2.imread(image_path)

    bottom = np.zeros(im.shape, np.uint8)
    bottom.fill(255)
    top = im.copy()
    overlapping = cv2.addWeighted(bottom, 0.2, top, 0.8, 0)

    point_size = 5
    point_color = (112, 25, 25)  # BGR
    thickness = -1
    point_list = peaks[li][0]
    for point in point_list:
        point = tuple(point)
        cv2.circle(overlapping, point, point_size, point_color, thickness)
    cv2.imwrite(save_name, overlapping)


def peaks_selection(votetype, F_l, image_size, image_path, save_path, image_name, visualize):
    peaks = []
    num_p = 20
    ori_peaks = []
    if votetype == 'gaussian':
        peaks_max = 100  # this is the key factor of the speed~
    else:
        peaks_max = 10000

    for li, p in enumerate(F_l):
        peaks.append([])
        ori_peaks.append([])
        area_para = 20
        for fi, p2 in enumerate(p):
            # p2 = np.resize(p2, (image_size[0], image_size[1]))
            peaks_l = select_peaks_l(p2, area_para, num_p)
            max_index = np.where(p2 == np.max(p2))
            ori_peaks.append(peaks_l)

            ratio_x = round(image_size[0] / p2.shape[0], 2)
            ratio_y = round(image_size[1] / p2.shape[1], 2)
            for i in range(len(peaks_l)):
                peaks_l[i][0] = int(peaks_l[i][0] * ratio_x)
                peaks_l[i][1] = int(peaks_l[i][1] * ratio_y)

            peaks[li].append(np.random.permutation(peaks_l)[:peaks_max])

        if visualize:
            show_peaks(peaks, li, image_path, image_name, save_path)

    return peaks, ori_peaks