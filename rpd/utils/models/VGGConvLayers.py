# -*- coding:utf-8 -*-
from torchvision.models.vgg import VGG, make_layers, cfgs, vgg16
import torch.utils.model_zoo as model_zoo
from torch import nn
import cv2
import torch
from torchvision import models, transforms


preprocess_transform = transforms.Compose([transforms.ToTensor()])

model_urls = {'vgg16': "https://download.pytorch.org/models/vgg16-397923af.pth"}


class Vgg16Layers(VGG):
    def __init__(self):
        # super(Vgg16Layers, self).__init__()
        super(Vgg16Layers, self).__init__(make_layers(cfgs['D']))

    def forward(self, x):
        convs_layers = []
        for l in self.features:
            x = l(x)
            if isinstance(l, nn.Conv2d):
                convs_layers.append(x)
        layers_outputs = []
        for i in [1, 3, 6, 9, 12]:
            layers_outputs.append(convs_layers[i])
        return [layers_outputs, convs_layers]


def vgg16_conv_layers():
    model = Vgg16Layers()
    model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model



# """test"""
# def feature_extraction(image):
#     """vgg16 feature extraction"""
#     # image = load_image(image_path)
#     dev = torch.device("cuda")
#     image = preprocess_transform(image).unsqueeze(0).to(dev)
#     image_size = image.squeeze().shape
#     image_size = tuple([image_size[1], image_size[2], image_size[0]])
#     dev = torch.device("cuda")
#     model = vgg16_conv_layers()
#     model.to(dev)
#     features = model(image)
#     # feature = features[-1].data.cpu().numpy()  #np.array.size:9216
#     feature = features[-1]  # torch.Size:[1, 4096]
#     return feature
#
# img = cv2.imread('/home/user/0-zoe_project/repeat_pattern_detection_v1_cleanup/rpd/input/test images/dot(5)_150_150_150_150.jpg')
# f = feature_extraction(img)