#!/usr/bin/python
#coding:utf-8

import os
from PIL import Image,ImageDraw

"""all the path info need to be filled when future use"""
path_imgs = '/' #ground_truth_path
for files in os.listdir(path_imgs):
    print(files)
    img_path = path_imgs+ files

    gt = img_path.split('_')

    image_name_not_yet = gt[-3].split('/')
    image_name = image_name_not_yet[-1]
    print(gt)
    print(image_name)
    gx = gt[-2]
    gy_not_yet = gt[-1].split('.')
    gy = gy_not_yet[0]
    print(gx)
    print(gy)

    start = [0, 0]

    img = Image.open(img_path)
    img = img.convert('RGB')

    image_size = img.size
    # print(image_size)
    # img_w = list(image_size)[0]
    # img_h = list(image_size)[1]

    img_d = ImageDraw.Draw(img)
    x_len, y_len = img.size

    if gx == 'x':
        gx = x_len
    if gy == 'x':
        gy = y_len

    x_step = int(gx)
    y_step = int(gy)

    for x in range(start[0], x_len, x_step):
        img_d.line(((x, 0), (x, y_len)), fill=(255, 0, 0), width=2)
    for y in range(start[1], y_len, y_step):
        j = y_len - y - 1
        img_d.line(((0, j), (x_len, j)), fill=(255, 0, 0), width=2)
    img.show()

    if not os.path.exists('./'):
        os.mkdir('./')
    img.save("./" + image_name + '_g.png')






