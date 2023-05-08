#!/usr/bin/python
#coding:utf-8

import xlrd
import os
from PIL import Image, ImageDraw, ImageFilter
import imghdr


def png2jpg(file_path):
    """change png to jpg"""
    files = os.listdir(file_path)
    for file in files:
        if file.endswith('png'):
            src = os.path.join(file_path, file)
            r_name = file.split('.')[0] + '.jpg'
            dct = os.path.join(file_path, r_name)
            os.rename(src, dct)


def read_excel(excelname, rename_path, save_path):
    xlsx = xlrd.open_workbook(excelname)
    sheet1 = xlsx.sheets()[1]    # 获得第1张sheet，索引从0开始
    sheet1_name = sheet1.name    # 获得名称
    sheet1_cols = sheet1.ncols   # 获得列数
    sheet1_nrows = sheet1.nrows  # 获得行数
    start = [0, 0]

    for i in range(2, sheet1_nrows):
        # try:
            # sheet1_nrows4 = sheet1.row_values(4)  # 获得第4行数据
            # sheet1_cols2 = sheet1.col_values(2)   # 获得第2列数据
            # cell23 = sheet1.row(2)[3].value       # 查看第3行第4列数据

            files = str(sheet1.row(i)[0].value)
            # print 'image_name: ' + files

            dx = sheet1.row(i)[1].value
            if dx != 'x':
                dx = int(dx)
            else:
                dx = str(dx)

            dy = sheet1.row(i)[2].value
            if dy != 'x':
                dy = int(dy)
            else:
                dy = str(dy)

            dstar = [dx, dy]
            # print dstar

            image_name = 'input/texture images/' + files + '.jpg'
            if os.path.exists(image_name):
                img_0 = Image.open(image_name)
                img_0 = img_0.convert('RGB')
                img_0.save(rename_path + files + '_' + str(dx) + '_' + str(dy) + '.jpg')
                img = img_0.copy()
                img_d = ImageDraw.Draw(img)
                x_len, y_len = img.size
                if dstar[0] != 'x':
                    for x in range(start[0], x_len, dstar[0]):
                        img_d.line(((x, 0), (x, y_len)), fill=(255, 0, 0), width=2)
                if dstar[1] != 'x':
                    for y in range(start[1], y_len, dstar[1]):
                        img_d.line(((0, y), (x_len, y)), fill=(255, 0, 0), width=2)
                # img.show()
                img.save(save_path + files + '_' + str(dx) + '_' + str(dy) + '.png')
            else:
                print files

        # except Exception as e:
        #     print("Erro" + image_name)
        # continue


def main():
    save_path = '/home/user/0-zoe_project/repeat_pattern_detection_v1_cleanup/rpd/input/Ground_truth/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    rename_path = '/home/user/0-zoe_project/repeat_pattern_detection_v1_cleanup/rpd/input/rename_images/'
    if not os.path.exists(rename_path):
        os.makedirs(rename_path)

    png2jpg(file_path="/home/user/0-zoe_project/repeat_pattern_detection_v1_cleanup/rpd/input/texture images/")
    ground_truth_path = '/home/user/0-zoe_project/repeat_pattern_detection_v1_cleanup/rpd/input/ground_truth.xlsx'
    read_excel(excelname=ground_truth_path, rename_path=rename_path, save_path=save_path)

if __name__ == "__main__":
    main()
    print '====over===='
