# -*- coding:utf-8 -*-
import numpy as np
import cv2
from scipy.ndimage.filters import maximum_filter, gaussian_filter
from scipy.stats import multivariate_normal


def select_filters(conv_feats):
    F_l = []
    mean_l = []
    delta = 0.65
    sigma_l = []  # # they use a fixed Gaussian filter = [13,13]

    for li, l in enumerate(conv_feats):
        mean_l.append([])
        maps = l.squeeze().detach().cpu().numpy()
        for fi, fmap in enumerate(maps):
            mu_fl = np.mean(fmap)
            mean_l[li].append(mu_fl)

    for li, l in enumerate(conv_feats):
        F_l.append([])
        max_l = np.max(mean_l[li])
        maps = l.squeeze().detach().cpu().numpy()
        for maps_num in range(len(mean_l[li])):
            mean_map = mean_l[li][maps_num]
            if mean_map > delta * max_l:
                F_l[li].append(maps[maps_num])
    return F_l


def select_peaks(F_l, image_size):
    peaks = []
    for li, selected_maps in enumerate(F_l):
        peaks.append([])
        for fi, fmap in enumerate(selected_maps):
            mu_fl = np.mean(fmap)
            std_fl = np.std(fmap)
            ratio_x = image_size[0] / fmap.shape[0]
            ratio_y = image_size[1] / fmap.shape[1]
            a = 2 * std_fl + mu_fl
            selected_fp = np.where(fmap >= a)
            x = selected_fp[0].reshape(-1, 1) * ratio_x
            y = selected_fp[1].reshape(-1, 1) * ratio_y
            peaks_l = np.hstack((x, y))
            peaks[li].append(peaks_l)
    return peaks


def displacement_com(peaks, image_size):
    disps = []
    dstars=[]
    V = []
    dim_a = [image_size[0] / 20, image_size[1] / 20]

    for li, p in enumerate(peaks):
        disps.append([])
        for fi, p2 in enumerate(p):
            quant_r, quant_c = np.mgrid[0:image_size[0]:1, 0:image_size[1]:1]
            Vxy = np.zeros(quant_r.shape)
            quant_rc = np.empty(quant_r.shape + (2,), dtype=np.float32)
            quant_rc[:, :, 0] = quant_r
            quant_rc[:, :, 1] = quant_c

            p2_cor = np.where(p2[:, 0] > p2[:, 1])
            p2_selected = p2[p2_cor]
            pairs_inds = np.asarray(np.meshgrid(np.arange(p2_selected.shape[0]), np.arange(p2_selected.shape[0])),
                                    dtype=np.uint8).T.reshape(-1, 2)
            pairs_inds = pairs_inds[pairs_inds[:, 0] > pairs_inds[:, 1]]

            if pairs_inds.shape[0] > 0:
                tmp_disps = np.abs(p2[pairs_inds[:, 0]] - p2[pairs_inds[:, 1]])
            else:
                tmp_disps = ([[]])

            if len(tmp_disps) == False:
                continue

            for ij, dij in enumerate(tmp_disps):
                if len(dij) > 0:
                    tmp_Vfiij = multivariate_normal.pdf(quant_rc, mean=dij,
                                                        cov=np.asarray([[13, 0], [0, 13]], dtype=np.float32))
                    tmp_Vfiij /= tmp_disps.shape[0]
                    Vxy += tmp_Vfiij
            V.append(Vxy)

            starting_ind = 10
            dstar = np.asarray((Vxy[starting_ind:, 0].argmax() + starting_ind, Vxy[0, starting_ind:].argmax() + starting_ind))
            if dstar[0] > dim_a[0] and dstar[1] > dim_a[1]:
                dstars.append(dstar)

    return V, dstars


def get_dstar(V, image_size):
    """find best step h,w"""
    starting_ind = 10
    dstars = []
    dim_a = [image_size[0] / 20, image_size[1] / 20]
    for Vxy in V:
        dstar = np.asarray(
            (Vxy[starting_ind:, 0].argmax() + starting_ind, Vxy[0, starting_ind:].argmax() + starting_ind))
        if dstar[0] > dim_a[0] and dstar[1] > dim_a[1]:
            dstars.append(dstar)
    return dstars


