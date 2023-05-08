# -*- coding:utf-8 -*-
from PIL import Image, ImageDraw
import csv


def compute_result(image_name, img_x, img_y, dstar):
    tmp = image_name.split('_')
    gx = tmp[-2]
    gy = tmp[-1]
    if gx == 'x':
        gx = img_x
    if gy == 'x':
        gy = img_y
    gx = int(gx)
    gy = int(gy)
    dx = dstar[0]
    dy = dstar[1]
    i_x = float(min(gx, dx)*min(gy, dy))
    i_y = float(max(gx, dx)*max(gy, dy))
    iou = round(i_x/i_y, 4)
    g_n = round(img_x/gx, 1) * round(img_y/gy, 1)
    d_n = round(img_x/dx, 1) * round(img_y/dy, 1)
    prop = round(d_n/g_n, 4)
    return iou, prop


def save_res_csv(res_name, data, refine):
    with open(res_name, 'wb') as f:
        csv_write = csv.writer(f)
        if refine:
            csv_head = [
                        'image_name',
                        'filter selection(time_cost1)',
                        'peak selection(time_cost2)',
                        'voting process(time_cost3)',
                        'refine_time(time_cost4)',
                        'time_cost_all',
                        'dstar_bf',
                        'iou_bf',
                        'prop_bf',
                        'dstar',
                        'iou',
                        'prop',
                        ]
        else:
            csv_head = [
                        'image_name',
                        'filter selection(time_cost1)',
                        'peak selection(time_cost2)',
                        'voting process(time_cost3)',
                        'time_cost_all',
                        'dstar',
                        'iou',
                        'prop',
                        ]
        csv_write.writerow(csv_head)
        for i in data:
            csv_write.writerow(i)


def grid_draw(image_path, dstar, save_name):
    start = [0, 0]
    img = Image.open(image_path)
    img = img.convert('RGB')
    img_d = ImageDraw.Draw(img)
    x_len, y_len = img.size
    if dstar[0] != x_len:
        for x in range(start[0], x_len, dstar[0]):
            img_d.line(((x, 0), (x, y_len)), fill=(255, 0, 0), width=2)
    if dstar[1] != y_len:
        for y in range(start[1], y_len, dstar[1]):
            img_d.line(((0, y), (x_len, y)), fill=(255, 0, 0), width=2)
    # img.show()
    img.save(save_name)