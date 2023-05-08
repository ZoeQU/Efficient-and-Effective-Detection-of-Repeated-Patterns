# -*- coding:utf-8 -*-
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from custom_plot import custom_plot


# def near_regular_bk(selected_filters, peaks, dstar, img_path, savepath):
#     sigma_l = []
#     alfa_l = [5, 7, 15, 15, 15]
#     fi_prctile = 0.8
#     delta = 0.65
#     subsample_pairs = 1000
#
#     fi_acc = []
#     for li, p in enumerate(peaks):
#         for fi, p2 in enumerate(p):
#             pairs_inds = np.asarray(np.meshgrid(np.arange(p2.shape[0]), np.arange(p2.shape[0])),
#                                     dtype=np.uint8).T.reshape(-1, 2)
#             pairs_inds = pairs_inds[pairs_inds[:, 0] > pairs_inds[:, 1]]
#
#             if pairs_inds.shape[0] > 0:
#                 tmp_disps = np.abs(p2[pairs_inds[:, 0]] - p2[pairs_inds[:, 1]])
#
#             else:
#                 fi_acc.append(0)
#                 continue
#
#             tmp_disps = tmp_disps[np.random.permutation(tmp_disps.shape[0])[:subsample_pairs]]
#             fi_acc.append(len([1 for dij in tmp_disps if (np.linalg.norm(dij - dstar)) < 3 * alfa_l[li]]))
#
#     param_fi = np.percentile(fi_acc, fi_prctile)
#
#     # # find weights for filters
#     disps_star = []
#     weights = []
#
#     for li, p in enumerate(peaks):
#         disps_star.append([])
#         weights.append([])
#
#         for fi, p2 in enumerate(p):
#             pairs_inds = np.asarray(np.meshgrid(np.arange(p2.shape[0]), np.arange(p2.shape[0])),
#                                     dtype=np.uint8).T.reshape(-1, 2)
#             pairs_inds = pairs_inds[pairs_inds[:, 0] > pairs_inds[:, 1]]
#
#             if pairs_inds.shape[0] > 0:
#                 tmp_disps = np.abs(p2[pairs_inds[:, 0]] - p2[pairs_inds[:, 1]])
#
#             else:
#                 tmp_disps = np.asarray([[]])
#
#             weights[li].append(0)
#             if tmp_disps.size == 0:
#                 continue
#
#             tmp_disps = tmp_disps[np.random.permutation(tmp_disps.shape[0])[:subsample_pairs]]
#
#             for ij, dij in enumerate(tmp_disps):
#                 tmp_diff = np.linalg.norm(dij - dstar)
#                 if tmp_diff < 3 * alfa_l[li]:
#                     # φ è 80esimo percentile, bisogna sommare i pesi per calcolare per ogni filtro
#                     wijfl = np.exp(-(tmp_diff ** 2)
#                                    / (2 * (alfa_l[li] ** 2))) \
#                             / (tmp_disps.shape[0] + param_fi)
#                     weights[li][-1] += wijfl
#
#     # # accumulate origin coordinates loss
#     acc_origin = []
#     acc_origin_weights = []
#     for li, w in enumerate(weights):
#         for fi in selected_filters[li]:
#             p2 = peaks[li][0]
#             pairs_inds = np.asarray(np.meshgrid(np.arange(p2.shape[0]), np.arange(p2.shape[0])),
#                                     dtype=np.uint8).T.reshape(-1, 2)
#             pairs_inds = pairs_inds[pairs_inds[:, 0] > pairs_inds[:, 1]]
#
#             if pairs_inds.shape[0] > 0:
#                 tmp_disps = np.abs(p2[pairs_inds[:, 0]] - p2[pairs_inds[:, 1]])
#
#             else:
#                 # fi_acc.append(0)
#                 continue
#
#             cons_disps = [dij for ij, dij in enumerate(tmp_disps)
#                           if (np.linalg.norm(dij - dstar)) < 3 * alfa_l[li]]
#
#             cons_disps_weights = [np.exp(-(np.linalg.norm(dij - dstar) ** 2) /
#                                          (2 * (alfa_l[li] ** 2))) / (tmp_disps.shape[0] + param_fi)
#                                   for dij in cons_disps]
#
#             acc_origin.extend(cons_disps)
#             acc_origin_weights.extend(cons_disps_weights)
#
#     o_r = np.linspace(-dstar[0], dstar[0], 10)
#     o_c = np.linspace(-dstar[1], dstar[1], 10)
#
#     min_rc = (-1, -1)
#     min_val = np.inf
#     for r in o_r:
#         for c in o_c:
#             tmp_orig = np.asarray([r, c])
#             tmp_val = [np.linalg.norm(np.mod((dij - tmp_orig), dstar) - (dstar / 2)) * acc_origin_weights[ij]
#                        for ij, dij in enumerate(acc_origin)]
#             tmp_val = np.sum(tmp_val)
#             if tmp_val < min_val:
#                 min_val = tmp_val
#                 min_rc = (r, c)
#
#     boxes = []
#     tmp_img = np.array(Image.open(img_path))
#     for ri in range(100):
#         min_r = min_rc[0] + (dstar[0] * ri) - (dstar[1] / 2)
#         if min_r > tmp_img.shape[0]:
#             break
#         for ci in range(100):
#             min_c = min_rc[1] + (dstar[1] * ci) - dstar[0] / 2
#             if min_c > tmp_img.shape[1]:
#                 break
#             tmp_box = np.asarray([min_c, min_r, dstar[1], dstar[0]])
#             boxes.append(tmp_box)
#
#     img = Image.open(img_path).convert('RGB')
#     img_name = img_path.split('/')[-1][:-4]
#     savename = savepath + img_name + '_' + str(dstar) + '_near_regular.png'
#     custom_plot(img, boxes, None, save_path=savename)


def near_regular(peaks, dstar, img_path, savename):
    alfa_l = [1, 1, 1, 1, 1]
    param_fi = 0.8

    dstar = np.asarray(dstar)
    acc_origin = []
    acc_origin_weights = []

    for li, p in enumerate(peaks):
        for fi, p2 in enumerate(p):
            pairs_inds = np.asarray(np.meshgrid(np.arange(p2.shape[0]), np.arange(p2.shape[0])),
                                    dtype=np.uint8).T.reshape(-1, 2)
            pairs_inds = pairs_inds[pairs_inds[:, 0] > pairs_inds[:, 1]]

            if pairs_inds.shape[0] > 0:
                tmp_disps = np.abs(p2[pairs_inds[:, 0]] - p2[pairs_inds[:, 1]])

            else:
                tmp_disps = np.asarray([[]])

            cons_disps = [dij for ij, dij in enumerate(tmp_disps)
                          if (np.linalg.norm(dij - dstar)) < alfa_l[li] * 5]

            cons_disps_weights = [np.exp(-(np.linalg.norm(dij - dstar) ** 2) /
                                         (2 * (alfa_l[li] ** 2))) / (tmp_disps.shape[0] + param_fi)
                                  for dij in cons_disps]

            acc_origin.extend(cons_disps)
            acc_origin_weights.extend(cons_disps_weights)

    o_r = np.linspace(-dstar[0], dstar[0], 10)
    o_c = np.linspace(-dstar[1], dstar[1], 10)


    min_rc = (-1, -1)
    min_val = np.inf
    for r in o_r:
        for c in o_c:
            tmp_orig = np.asarray([r, c])
            tmp_val = [np.linalg.norm(np.mod((dij - tmp_orig), dstar) - (dstar / 2)) * acc_origin_weights[ij]
                       for ij, dij in enumerate(acc_origin)]

            tmp_val = np.sum(tmp_val)
            if tmp_val < min_val:
                min_val = tmp_val
                min_rc = (r, c)


    boxes = []
    centriods = []
    tmp_img = np.array(Image.open(img_path))
    for ri in range(100):
        min_r = min_rc[0] + (dstar[0] * ri) - (dstar[0] / 2)
        if min_r > tmp_img.shape[0]:
            break
        for ci in range(100):
            min_c = min_rc[1] + (dstar[1] * ci) - dstar[1] / 2
            if min_c > tmp_img.shape[1]:
                break
            tmp_box = np.asarray([min_c, min_r, dstar[1], dstar[0]])

            cx = int(dstar[1] / 2 + min_c)
            cy = int(dstar[0] / 2 + min_r)
            centriods.append([cx, cy])
            boxes.append(tmp_box)


    img = Image.open(img_path).convert('RGB')
    img_name = img_path.split('/')[-1][:-4]
    custom_plot(img, boxes, None, save_path=savename)

    c_savename = savename[:-4] + 'centriods.png'
    im = cv2.imread(img_path)
    bottom = np.zeros(im.shape, np.uint8)
    bottom.fill(255)
    top = im.copy()
    overlapping = cv2.addWeighted(bottom, 0.2, top, 0.8, 0)

    point_size = 5
    point_color = (112, 25, 25)  # (31, 23, 176)  # BGR
    thickness = -1
    for point in centriods:
        point = tuple(point)
        cv2.circle(overlapping, point, point_size, point_color, thickness)
    cv2.imwrite(c_savename, overlapping)