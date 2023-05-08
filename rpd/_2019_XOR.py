# !/usr/bin/python
# coding:utf-8

import numpy as np
from PIL import Image, ImageDraw
import os
import cv2
import time
from utils.functions.result_funs import compute_result, save_res_csv, grid_draw


def dstar_horizontal(img):
    dictionary = {}
    img_copy = img.copy()
    i_vertical = img.shape[0]  # vertical
    i_horizatal = img.shape[1]  # horizatal
    im = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
    main_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    for i in range(i_horizatal):
        dict = {}
        cyclic_image = np.roll(im, i+1, axis=1)  # roll left
        xor_image = cv2.bitwise_xor(cyclic_image, main_image)
        key = np.mean(xor_image)
        value = i
        dict[key] = value

        if key in dictionary:
            if isinstance(dictionary[key], list):
                thelist = dictionary[key]
                thelist.append(value)
                dictionary[key] = thelist
            else:
                thelist = dictionary[key]
                dictionary[key] = list()
                dictionary[key].append(thelist)
                dictionary[key].append(value)
        else:
            dictionary[key] = value

    min_avg = min(dictionary, key=dictionary.get)
    x_min_tmp = dictionary[min_avg]

    if isinstance(x_min_tmp, int):
        x_min_count = 1
    else:
        x_min_count = len(x_min_tmp)

    if x_min_count == 1:
        if dictionary[min_avg] == 1 or dictionary[min_avg] == i_horizatal-1:
            # print "No Any Repeat Pattern in Horizontal Direction"
            d = i_horizatal
            return d
        else:
            distance = dictionary[min_avg] + 1
            return distance
    else:
        x_min_1 = dictionary[min_avg][0]
        x_min_2 = dictionary[min_avg][1]
        distance = abs(int(x_min_1) - int(x_min_2))
        if distance == 1:
            # print 'Horizontal Repeating Pattern'
            d = i_horizatal
            return d
        else:
            return distance


def dstar_vertical(img):
    dictionary = {}
    img_copy = img.copy()
    i_vertical = img.shape[0]  # vertical
    i_horizatal = img.shape[1]  # horizatal
    im = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
    main_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    for i in range(i_vertical):
        dict = {}
        cyclic_image = np.roll(im, i+1, axis=0)   # roll down?
        xor_image = cv2.bitwise_xor(cyclic_image, main_image)
        key = np.mean(xor_image)
        value = i
        dict[key] = value

        if key in dictionary:
            if isinstance(dictionary[key], list):
                thelist = dictionary[key]
                thelist.append(value)
                dictionary[key] = thelist
            else:
                thelist = dictionary[key]
                dictionary[key] = list()
                dictionary[key].append(thelist)
                dictionary[key].append(value)
        else:
            dictionary[key] = value

    min_avg = min(dictionary, key=dictionary.get)
    x_min_tmp = dictionary[min_avg]

    if isinstance(x_min_tmp, int):
        x_min_count = 1
    else:
        x_min_count = len(x_min_tmp)

    if x_min_count == 1:
        if dictionary[min_avg] == 1 or dictionary[min_avg] == i_vertical-1:
            d = i_vertical
            # print "No Any Repeat Pattern in Vertical Direction"
            return d
        else:
            distance = dictionary[min_avg] + 1
            return distance
    else:
        x_min_1 = dictionary[min_avg][0]
        x_min_2 = dictionary[min_avg][1]
        distance = abs(int(x_min_1) - int(x_min_2))
        if distance == 1:
            # print 'Vertical Repeat Pattern'
            d = i_vertical
            return d
        else:
            return distance


def run(folder, refine, regular, visualize):
    # # 1. creat save path
    INPUT_FOLDER = 'input/textureimagesallbyfolder/' + str(folder) + '/'
    save_path = 'output/2019_XOR_grid/' + str(folder) + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    res_name = save_path + folder + '_results.csv'

    # # 2. load images
    datas = []
    for files in os.listdir(INPUT_FOLDER):
        try:
            image_name = INPUT_FOLDER + files
            im = cv2.imread(image_name)
            x_len = im.shape[1]
            y_len = im.shape[0]
            start_time = time.time()

            # # 3. compute dstar_horizaontal
            distance_horizonal = dstar_horizontal(im)

            # # 4. compute dstar_vertical
            distance_vertical = dstar_vertical(im)

            # # 5. dstar and draw results
            dstar = [distance_horizonal, distance_vertical]
            save_name = save_path + files[:-4] + '_2019XOR.png'
            grid_draw(image_name, dstar, save_name)

            # # 6. compute time cost and save results
            time_cost_all = round(time.time() - start_time, 4)
            iou, prop = compute_result(image_name=files[:-4], img_x=x_len, img_y=y_len, dstar=dstar)
            data = [files[:-4], '', '', '', time_cost_all, dstar, iou, prop]
            datas.append(data)

        except Exception as e:
            with open("logs/2019_XOR_erro.txt", "a") as f:
                f.write(str(files) + '\n')
                f.write(str(e) + '\n')
        continue

    save_res_csv(res_name, datas, refine=False)
    print ('===finish===')


run(folder='texture1', refine=False, regular=False, visualize=False)