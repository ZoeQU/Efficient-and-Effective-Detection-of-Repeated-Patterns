# -*- coding:utf-8 -*-
from PIL import Image, ImageFilter
import numpy as np
import cv2
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


def color_cluster(im, colornum):
    temp_im = np.array(im, dtype=np.float64) / 255
    w, h, c = temp_im.shape
    dd = temp_im.reshape(w * h, c)
    Kcolor = KMeans(n_clusters=colornum)
    Kcolor.fit(dd)
    label_pred = Kcolor.predict(dd)
    cluster_colors = Kcolor.cluster_centers_
    # print(cluster_colors)
    img2 = temp_im.copy()
    for i in range(w):
        for j in range(h):
            ij = i * h + j
            img2[i][j] = cluster_colors[label_pred[ij]]

    return cluster_colors, img2


def color_blocks(savename, cluster_colors):
    color_fig = plt.figure()
    box = color_fig.add_subplot(111, aspect='equal')
    for i in range(len(cluster_colors)):
        face_color = tuple(cluster_colors[i])
        loc_x = i * 0.2
        Loc = tuple([loc_x, 0])
        tmp_box = plt.Rectangle(Loc, 0.2, 0.8, facecolor=face_color,edgecolor='r',fill=True)
        box.add_patch(tmp_box)
    plt.axis('off')
    color_fig.savefig(savename, dpi=90, bbox_inches='tight')
    # plt.show()
    plt.close()


def pre_process(image_name, colornum, savename):
    # 1. open the image
    image_names = []
    im = Image.open(image_name)
    im = im.convert('RGB')
    wa, ha = im.size
    im = im.filter(ImageFilter.SMOOTH_MORE)

    # 2. kmeans for color reduction
    cluster_colors, img2 = color_cluster(im, colornum)

    # 3. visualization color blocks
    # TODO(zoe) add savename
    savenametemp = ''
    color_blocks(savenametemp, cluster_colors)

    # 4. color separation
    image_colors = []
    # TODO(zoe) savenamehead
    savenamehead = ''
    for i in range(len(cluster_colors)):
        imagecopy = img2.copy()
        for y in range(ha):
            for x in range(wa):
                if (imagecopy[y, x][0] != cluster_colors[i][0] and
                        imagecopy[y, x][1] != cluster_colors[i][1] and
                        imagecopy[y, x][2] != cluster_colors[i][2]):
                    imagecopy[y, x] = (1 - cluster_colors[i][0],
                                       1 - cluster_colors[i][1],
                                       1 - cluster_colors[i][2])
        ic_a = imagecopy[:, :, ::-1]
        ic_a2 = np.rint(ic_a * 255)
        image_colors.append(ic_a2)
        plt.imshow(ic_a)
        plt.axis('off')
        plt.show()
        image_name_by_color = savenamehead + '_' + str(i) + '_color_group.png'
        # TODO(zoe) cv2.imwrite and plt.savefigure rgb or bgr  any diff?
        cv2.imwrite(image_name_by_color, ic_a2)
        # cv2.imshow()
        image_names.append(image_name_by_color)
        # plt.savefig(image_name_by_color,bbox_inches='tight',pad_inches=0.0)
        # plt.savefig("grid_draw_0608//" + image_name[15:] + '_' + str(i) + '_color_group.png',bbox_inches='tight',pad_inches=0.0)
        # plt.show()

        # else:
        #     image_names.append(image_name)

    # TODO(zoe): this part not finish.
    disps = []
    for idx in range(len(image_names)):
        img_path = image_names[idx]
        im = Image.open(img_path)
        im = im.convert('RGB')
        # im.show()
        in_ = np.array(im, dtype=np.float64)

        in2_ = in_.transpose((2, 0, 1))

        # # =======try another boundary=======
        # boundary_image = cv2.imread(image_name)
        # fuse = cv2.Canny(boundary_image, 400, 400)
        # plt.imshow(fuse, cmap='gray')
        # # plt.imshow(boundary_image_resize) #dark blue background, and yeallow bundary
        # plt.show()

        """BDCN edge detection"""
        BDCN_path = savenamehead + '_bdcn_boundary.png'
        # fuse = BDCN_test(img_path)
        fuse = 255 * BDCN_test(img_path)
        plt.imshow(fuse, cmap='gray')
        plt.axis('off')
        plt.show()
        fuse_save = 255 - fuse
        # custom_blur_demo(fuse_save)
        # cv2.medianBlur(fuse_save, 5)
        cv2.imwrite(BDCN_path, fuse_save)

        """Alexnet"""
        image = load_image(img_path)
        image_size = image.squeeze().shape
        image_size = tuple([image_size[1], image_size[2], image_size[0]])
        dev = torch.device("cuda")
        model = alexnet_conv_layers()
        model.to(dev)
        filters = model(image)

        """filter selection"""
        temp_disps, sigma_l = boundray_com(filters_boundray=fuse, filters=filters, img_path=img_path,image_size=image_size)
        disps.append(temp_disps)

    return disps, sigma_l, image_size

# """old version has color reduction"""
# disps, sigma_l,image_size = pre_process(image_path)
# V = new_vote(image_size, disps)
# # show activated peaks, peaks => V
# plt.imshow(V)
# plt.axis('off')
# plt.savefig("grid_draw_0608//" +image_name[16:-10] + '_2019activation peaks.png')
# plt.show()