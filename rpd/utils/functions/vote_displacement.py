# -*- coding:utf-8 -*-
import numpy as np
import cv2
from scipy.ndimage.filters import maximum_filter, gaussian_filter
from scipy.stats import multivariate_normal


def vote_displacement(peaks, image_size, sigma_l):
    quant_r, quant_c = np.mgrid[0:image_size[0]:1, 0:image_size[1]:1]
    V = np.zeros(quant_r.shape)
    quant_rc = np.empty(quant_r.shape + (2,), dtype=np.float32)
    quant_rc[:, :, 0] = quant_r
    quant_rc[:, :, 1] = quant_c
    disps = []
    # subsample_pairs = 10
    disp_min = [image_size[0] / 20, image_size[1] / 20]
    for li, p in enumerate(peaks):
        disps.append([])
        for fi, p2 in enumerate(p):
            if len(p2):
                p2 = np.array(p2)
                pairs_inds = np.asarray(np.meshgrid(np.arange(p2.shape[0]), np.arange(p2.shape[0])),
                                        dtype=np.uint8).T.reshape(-1, 2)
                pairs_inds = pairs_inds[pairs_inds[:, 0] > pairs_inds[:, 1]]
                if pairs_inds.shape[0] > 0:
                    tmp_disps = np.abs(p2[pairs_inds[:, 0]] - p2[pairs_inds[:, 1]])
                else:
                    tmp_disps = []

                disps[li].append(tmp_disps)

                if len(tmp_disps)==False:
                    continue

                for ij, dij in enumerate(tmp_disps):
                    if dij[0] > disp_min[0] and dij[1] > disp_min[1]:
                        tmp_Vfiij = multivariate_normal.pdf(quant_rc, mean=dij
                                                            , cov=np.asarray([[sigma_l[li], 0], [0, sigma_l[li]]]
                                                                             , dtype=np.float32))
                        tmp_Vfiij /= tmp_disps.shape[0]
                        V += tmp_Vfiij
            else:
                disps[li].append([])

    return disps, V