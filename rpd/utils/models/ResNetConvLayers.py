# -*- coding:utf-8 -*-
from torchvision.models.resnet import resnet50, ResNet, Bottleneck
import torch.utils.model_zoo as model_zoo
from torch import nn
import cv2
import torch
from torchvision import models, transforms

preprocess_transform = transforms.Compose([transforms.ToTensor()])

model_urls = {'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'}


class ResNetConvLayers(nn.Module):
    def __init__(self):
        super(ResNetConvLayers, self).__init__()
        self.model = models.resnet50(pretrained=True)

    def forward(self, x):
        layers_outputs = []
        x = self.model.conv1(x)
        layers_outputs.append(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        layers_outputs.append(x)
        x = self.model.layer2(x)
        layers_outputs.append(x)
        x = self.model.layer3(x)
        layers_outputs.append(x)
        x = self.model.layer4(x)
        layers_outputs.append(x)
        return layers_outputs


def resnet50_conv_layers():
    model = ResNetConvLayers()
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
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
#     model = ResNetConvLayers()
#     model.to(dev)
#     features = model(image)
#     # feature = features[-1].data.cpu().numpy()
#     feature = features[-1]
#     return feature
#
# img = cv2.imread('/home/user/0-zoe_project/repeat_pattern_detection_v1_cleanup/rpd/input/test images/dot(5)_150_150_150_150.jpg')
# f = feature_extraction(img)