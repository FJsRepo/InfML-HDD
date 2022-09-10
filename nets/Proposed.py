import torch
import torch.nn as nn
from .utils import load_state_dict_from_url
import numpy as np
import cv2
import matplotlib.pyplot as plt


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=5, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #stride = 1
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_classes, num_classes)
        self.fc_x = nn.Linear(512, num_classes)
        self.fc_last = nn.Linear(num_classes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def Draw_Row_Average(self, row, index2, stage, threshold):
        X = np.arange(0, len(row), 1)  # 0-287
        Y = row[X]
        Y_1 = np.full(len(row), threshold)
        plt.plot(X, Y)
        plt.plot(X, Y_1, color='red',linewidth=2)
        plt.xlabel('Row Sequence')
        plt.ylabel('Average Row Grayscale')
        # plt.ylim(-60, 60)
        plt.ylim(0, 255)
        # plt.show()
        plt.savefig('../HorizonNet/images/tire{:05d}_{}_RowAverage.png'.format(index2, stage), dpi=1000)
        plt.close()

    def get_stage_images(self, stage, x, threshold, num_classes=5):
        temp_x = x
        temp_x_numpy = temp_x.detach().cpu().numpy()
        batch_size = temp_x_numpy.shape[0]
        tires = temp_x_numpy.shape[1]
        left_temp = []
        right_temp = []
        left_points = np.zeros(batch_size)
        right_points = np.zeros(batch_size)
        for index1 in range(batch_size):
            for index2 in range(tires):
                if temp_x_numpy[index1, index2, :, :].max() != 0:
                    temp_img = temp_x_numpy[index1, index2, :, :]
                    img_h, img_w = temp_img.shape # img_h 72   img_w 96
                    temp_img = temp_img * (255 / (temp_img.max()))
                    _, temp_img = cv2.threshold(temp_img, 30, 255, cv2.THRESH_BINARY)
                    temp_img = temp_img.astype(np.uint8)
                    for index3 in range(3):
                        temp_img[index3, :] = temp_img[3+index3, :]
                        temp_img[:, index3] = temp_img[:, 3+index3]
                    for index4 in range(3):
                        temp_img[img_h-1-index4, :] = 0
                        temp_img[:, img_w - 1 - index4] = 0
                    row_average = (np.sum(temp_img, axis=1))/img_w
                    # plt.imshow(temp_img, cmap='gray')
                    # # plt.show()
                    # plt.savefig('../HorizonNet/images_Binary/tire{:05d}_{}.png'.format(index2, stage), dpi=1000)
                    # plt.close()
                    # self.Draw_Row_Average(row_average, index2, stage, threshold)
                    row_number = []
                    for index4 in range(img_h):
                        if row_average[index4] > threshold:
                            row_number.append(index4)
                    if len(row_number) >= 2:
                        if (sum(temp_img[row_number[0], :int(img_w/2)]) > sum(temp_img[row_number[0], int(img_w/2):])):
                            left_temp.append(row_number[0]/img_h)
                            right_temp.append(row_number[1]/img_h)
                        else:
                            left_temp.append(row_number[1]/img_h)
                            right_temp.append(row_number[0]/img_h)
                    elif len(row_number) == 1:
                        left_temp.append(row_number[0]/img_h)
                        right_temp.append(row_number[0]/img_h)

            left_temp.sort()
            right_temp.sort()

            if len(left_temp) != 0:
                left_points[index1] = left_temp[3*(len(left_temp)//4)]
            else:
                left_points[index1] = 0
            if len(right_temp) != 0:
                right_points[index1] = right_temp[3*(len(right_temp)//4)]
            else:
                right_points[index1] = 0

            if len(left_temp) == 0 and len(right_temp) != 0:
                left_points[index1] = right_points[index1]
            elif len(left_temp) != 0 and len(right_temp) == 0:
                right_points[index1] = left_points[index1]

            left_temp = []
            right_temp = []
        endpoints = np.zeros((batch_size, num_classes))
        for i in range(batch_size):
            endpoints[i, 1] = left_points[i]
            endpoints[i, 2] = right_points[i]
        endpoints = torch.tensor(endpoints, dtype=torch.float32)
        endpoints = endpoints.cuda()
        endpoints = self.relu(endpoints)
        return endpoints

    def lapalce(self, x):
        laplace_filter = np.array([
            [1, 1, 1],
            [1, -8.3, 1],
            [1, 1, 1],
        ])
        x_temp = x.detach().cpu().numpy()
        batch_size, image_num, _, _ = x_temp.shape
        for index in range(batch_size):
            for index2 in range(image_num):
                x_temp[index][index2] = cv2.GaussianBlur(x_temp[index][index2], (3, 3), 0)
                x_temp[index][index2] = cv2.filter2D(x_temp[index][index2], -1, laplace_filter)
                if sum(np.sum(x_temp[index][index2], axis=1)[2:5]) > 0:
                    x_temp[index][index2] = -x_temp[index][index2]
                x_temp[index][index2] = (x_temp[index][index2] + abs(x_temp[index][index2]))/2
        x_temp_tensor = torch.from_numpy(x_temp.copy())
        x_temp_tensor = x_temp_tensor.cuda()
        x_temp_tensor = self.relu(x_temp_tensor)
        return x_temp_tensor

    def _forward_impl(self, x):
        # temp_x = x
        # temp_x_numpy = temp_x.detach().cpu().numpy()
        # temp_x_numpy = temp_x_numpy[0, 0, :, :]
        # plt.imshow(temp_x_numpy, cmap='gray')
        # plt.savefig('../HorizonNet/images/original.png', dpi=1000)
        # plt.close()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.lapalce(x)

        endpoints = self.get_stage_images('After_conv1_maxpool', x, 100)

        # x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_x(x)
        endpoints = self.fc_last(endpoints)
        x = x/4 + 3*endpoints/4
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def init_laplace_filter_resnet18_34(model, layer):
    laplace_filter = np.array([
        [1, 1, 1],
        [1, -9, 1],
        [1, 1, 1],
    ])
    laplace_filter_tensor = torch.from_numpy(laplace_filter.copy())

    for key in model.state_dict().keys():
        if (layer in key) and ('conv2' in key):
        # if (layer in key) and ('conv1' in key or 'conv2' in key):
            print(key, "initial finished!")
            number_of_kernel, number_of_kernel_layers, _, _ = model.state_dict()[key].shape
            for index in range(number_of_kernel):
                for index2 in range(number_of_kernel_layers):
                    model.state_dict()[key][index, index2] = laplace_filter_tensor
    return model

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    print("-------Proposed--------")
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
