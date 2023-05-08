# -*- coding:utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import colors
from visualization import sim_curve


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    if sigma != 0:
        return (data - mu) / sigma, mu
    else:
        return data, mu


def crop_x(img, dstar):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = img_gray[:dstar[1], :dstar[0]]
    img_x = img_gray[:dstar[1], int(dstar[0] / 3):]
    return template, img_x


def refine(img, template, dstar, similarity):
    resx = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    resx = np.reshape(resx, (1, -1))
    mu = np.average(resx, axis=1)
    threshold = np.max(resx) * similarity
    loc_x = np.where(resx >= threshold)

    if len(loc_x[1]) < 2:
        threshold_w = np.max(resx)
        loc_x = np.where(resx >= threshold_w)
        dstar_x = loc_x[1]
        dstar_w = dstar_x + int(dstar[0] / 3)
        return dstar_w, mu, resx

    else:
        for ii in range(len(loc_x[1]) - 1):
            if loc_x[1][ii + 1] - loc_x[1][ii] < 2:
                dstar_x = loc_x[1][ii + 1]
                dstar_w = dstar_x + int(dstar[0] / 3)
                return dstar_w, mu, resx

            else:
                dstar_x = loc_x[1][ii]
                if dstar_x < 10:
                    continue
                dstar_w = dstar_x + int(dstar[0] / 3)
                return dstar_w, mu, resx


def crop_y(img, dstar):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = img_gray[:dstar[0], :dstar[1]]
    img_y = img_gray[int(dstar[0] / 3):, : dstar[1]]
    return template, img_y


def hsm(image_path, dstar, image_name, save_path, visualize):
    img = cv2.imread(image_path)
    i_h = img.shape[0]
    i_w = img.shape[1]

    # # 1. refine x-axis
    similarity = 0.87

    template_x, img_x = crop_x(img, dstar)
    template_y, img_y = crop_y(img, dstar)
    dstar_x, mux, resx = refine(img_x, template_x, dstar, similarity)

    i = 0
    if i < 5 and dstar_x > 0.55 * i_w:
        similarity = similarity - 0.01
        dstar_x, mux, resx = refine(img_x, template_x, dstar, similarity)
        i += 1

    if visualize:
        savename = save_path + image_name + '_x.png'
        title = 'similarity in x-axis'
        sim_curve(resx, savename, title)

    # # 2. refine y-axis
    dstar_y, muy, resy = refine(img_y, template_y, dstar, similarity)

    j = 0
    if j < 5 and dstar_y > 0.55 * i_h:
        similarity = similarity - 0.01
        dstar_y, muy, resy = refine(img_y, template_y, dstar, similarity)
        j += 1

    if visualize:
        savename = save_path + image_name + '_y.png'
        title = 'similarity in y-axis'
        sim_curve(resy, savename, title)

    dstar = [dstar_x, dstar_y]
    return dstar

