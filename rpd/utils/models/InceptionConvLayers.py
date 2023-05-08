# -*- coding:utf-8 -*-
# -*- coding:utf-8 -*-
from torchvision.models import Inception3
import torch.utils.model_zoo as model_zoo
import cv2
import torch
import torch.nn.functional as F
from torchvision import models, transforms

preprocess_transform = transforms.Compose([transforms.ToTensor()])

model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}


class InceptionConvLayers(Inception3):
    def __init__(self):
        super(InceptionConvLayers, self).__init__()
        self.model = models.inception_v3(pretrained=True)

    def forward(self, x):
        layers_outputs = []
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        layers_outputs.append(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        layers_outputs.append(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        layers_outputs.append(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        layers_outputs.append(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        layers_outputs.append(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        layers_outputs.append(x)
        # N x 768 x 17 x 17
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        layers_outputs.append(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=self.training)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)

        return layers_outputs


def inceptionv3_conv_layers():
    # features:
    # [Conv2d_1a_3x3(x), Conv2d_2b_3x3(x), Conv2d_3b_1x1(x),
    # Conv2d_4a_3x3(x), Mixed_5d(x), Mixed_6e(x), Mixed_7c(x)]
    model = InceptionConvLayers()
    model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']))
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
#     model = InceptionConvLayers()
#     model.to(dev)
#     features = model(image)
#     # feature = features[-1].data.cpu().numpy()
#     feature = features[-1]
#     return feature
#
# img = cv2.imread('/home/user/0-zoe_project/repeat_pattern_detection_v1_cleanup/rpd/input/test images/dot(5)_150_150_150_150.jpg')
# f = feature_extraction(img)