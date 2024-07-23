import os
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
import torch.nn.functional as F
import math
import copy



import torch
import torch.nn as nn
import random

import warnings
from torchvision import models
from efficientnet_pytorch import EfficientNet
from torchvision import transforms as trans
class GAMMA_sub1_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_transforms,
        oct_transforms,
        dataset_root,
        label_file="",
        filelists=None,
        num_classes=3,
        mode="train",
    ):
        self.dataset_root = dataset_root
        self.img_transforms = img_transforms
        self.oct_transforms = oct_transforms
        self.mode = mode.lower()
        self.num_classes = num_classes
        # 如果是加载训练集，则需要加载label
        if self.mode == "train":
            # 使用pandas读取label，label是one-hot形式
            label = {
                row["data"]: row[1:].values
                for _, row in pd.read_excel(label_file).iterrows()
            }

            self.file_list = [
                [f, label[int(f)]] for f in os.listdir(dataset_root) if f != '.DS_Store'
            ]
        # 如果是加载测试集，则label为空
        elif self.mode == "test":
            self.file_list = [[f, None] for f in os.listdir(dataset_root)]

        # 如果指定了加载哪些数据，则只加载指定的数据
        if filelists is not None:
            self.file_list = [
                item for item in self.file_list if item[0] in filelists
            ]

    def __getitem__(self, idx):
        # 获取指定下标的训练集和标签，real_index是样本所在的文件夹名称
        real_index, label = self.file_list[idx]
        # 彩色眼底图像的路径
        fundus_img_path = os.path.join(
            self.dataset_root, real_index, real_index + ".jpg"
        )
        # 光学相干层析(OCT)图片的路径集，一个3D OCT图片包含256张二维切片
        oct_series_list = sorted(
            os.listdir(os.path.join(self.dataset_root, real_index, real_index)),
            key=lambda s: int(s.split("_")[0]),
        )
        # 使用opencv读取图片，并转换通道 BGR -> RGB
        fundus_img = cv2.imread(fundus_img_path)[:, :, ::-1]
        # 读取3D OCT图片的一个切片，注意是灰度图 cv2.IMREAD_GRAYSCALE
        oct_series_0 = cv2.imread(
            os.path.join(
                self.dataset_root, real_index, real_index, oct_series_list[0]
            ),
            cv2.IMREAD_GRAYSCALE,
        )
        oct_img = np.zeros(
            (
                oct_series_0.shape[0],
                oct_series_0.shape[1],
                len(oct_series_list),
            ),
            dtype="uint8",
        )

        # 依次读取每一个切片
        for k, p in enumerate(oct_series_list):
            oct_img[:, :, k] = cv2.imread(
                os.path.join(self.dataset_root, real_index, real_index, p),
                cv2.IMREAD_GRAYSCALE,
            )
        # breakpoint()
        # 对彩色眼底图片进行数据增强
        if self.img_transforms is not None:
        #     from PIL import Image
        #     breakpoint()
        #     # if isinstance(fundus_img, np.ndarray):
        #     #     fundus_img = Image.fromarray(fundus_img)
        #     fundus_img = Image.fromarray(fundus_img)
            fundus_img = self.img_transforms(fundus_img)
        #     fundus_img =  np.array(fundus_img)

        # 对3D OCT图片进行数据增强
        if self.oct_transforms is not None:
        #     breakpoint()
        #     # from PIL import Image
        #     # if isinstance(oct_img, np.ndarray):
        #     # oct_img = Image.fromarray(oct_img)
            oct_img = self.oct_transforms(oct_img)
        #     # oct_img = np.array(oct_img)

        # 交换维度，变为[通道数，高，宽],  H, W, C -> C, H, W

        # fundus_img = fundus_img.transpose(2, 0, 1)
        # oct_img = oct_img.transpose(2, 0, 1)
        fundus_img = fundus_img.transpose(2, 0, 1).copy()
        oct_img = oct_img.transpose(2, 0, 1).copy()

        if self.mode == "test":
            return fundus_img, oct_img, real_index
        if self.mode == "train":
            label = label.argmax()
            return fundus_img, oct_img, label

    # 获取数据集总的长度
    def __len__(self):
        return len(self.file_list)


image_size = [256, 256]
# 三维OCT图片每个切片的大小
oct_img_size = [512, 512]

def random_horizontal_flip(image, p=0.5):
    if random.random() < p:  # 以p概率执行水平翻转
        return cv2.flip(image, 1)
    else:
        return image  # 不执行翻转，直接返回原图


def random_vertical_flip(image, p=0.5):
    if random.random() < p:  # 以p概率执行水平翻转
        return cv2.flip(image, 0)
    else:
        return image  # 不执行翻转，直接返回原图

from skimage.transform import rotate

def random_rotation(image, angle_range=(-10, 10), p=0.5):
    if random.random() < p:
        angle = np.random.uniform(angle_range[0], angle_range[1])
        rotated = rotate(image, angle, mode='reflect', preserve_range=True)
        return rotated.astype(np.uint8)
    else:
        return image




def random_crop_and_resize(image, scale_range=(0.8, 1.0), output_shape=(224, 224)):
    height, width = image.shape[:2]
    scale = np.random.uniform(scale_range[0], scale_range[1])
    new_height, new_width = int(height * scale), int(width * scale)

    y = np.random.randint(0, height - new_height + 1)
    x = np.random.randint(0, width - new_width + 1)

    cropped = image[y:y + new_height, x:x + new_width]
    resized = cv2.resize(cropped, output_shape)
    return resized


def augment_image(image):
    image = random_horizontal_flip(image)
    image = random_vertical_flip(image)
    image = random_rotation(image)
    # image = adjust_brightness(image)
    # image = adjust_contrast(image)
    image = random_crop_and_resize(image)
    return image
img_train_transforms = augment_image
oct_train_transforms = augment_image
img_val_transforms =augment_image
oct_val_transforms = augment_image

import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict
from typing import Optional, Callable

def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


"""

"""


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf

    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


"""

"""


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


"""
定义BN和激活函数
"""


class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,  # BN结构，默认为None
                 activation_layer: Optional[Callable[..., nn.Module]] = None):  # ac结构，默认为None
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            """
            不指定norm的话，默认BN结构
            """
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            """
            如果不指定激活函数，则默认SiLU激活函数
            """
            activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)激活函数使用的是SiLU，在版本1.70以上才有，建议1.7.1版本

        # 定义层结构[in_channels,out_channels,kernel_size,stride,padding,groups,bias=False],-->BN-->ac
        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),
                                               norm_layer(out_planes),
                                               activation_layer())


"""
定义SE模块
"""


class SqueezeExcitation(nn.Module):
    def __init__(self,
                 input_c: int,  # block input channel
                 expand_c: int,  # block expand channel
                 squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = input_c // squeeze_factor
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)
        self.ac1 = nn.SiLU()  # alias Swish
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x) :
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x


"""

"""


class InvertedResidualConfig:
    # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate
    def __init__(self,
                 kernel: int,  # 3 or 5
                 input_c: int,
                 out_c: int,
                 expanded_ratio: int,  # 1 or 6
                 stride: int,  # 1 or 2
                 use_se: bool,  # True
                 drop_rate: float,
                 index: str,  # 1a, 2a, 2b, ...
                 width_coefficient: float):
        self.input_c = self.adjust_channels(input_c, width_coefficient)
        self.kernel = kernel
        self.expanded_c = self.input_c * expanded_ratio
        self.out_c = self.adjust_channels(out_c, width_coefficient)
        self.use_se = use_se
        self.stride = stride
        self.drop_rate = drop_rate
        self.index = index

    @staticmethod
    def adjust_channels(channels: int, width_coefficient: float):
        return _make_divisible(channels * width_coefficient, 8)


"""

"""


class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        layers = OrderedDict()
        activation_layer = nn.SiLU  # alias Swish

        # expand
        if cnf.expanded_c != cnf.input_c:
            layers.update({"expand_conv": ConvBNActivation(cnf.input_c,
                                                           cnf.expanded_c,
                                                           kernel_size=1,
                                                           norm_layer=norm_layer,
                                                           activation_layer=activation_layer)})

        # depthwise
        layers.update({"dwconv": ConvBNActivation(cnf.expanded_c,
                                                  cnf.expanded_c,
                                                  kernel_size=cnf.kernel,
                                                  stride=cnf.stride,
                                                  groups=cnf.expanded_c,
                                                  norm_layer=norm_layer,
                                                  activation_layer=activation_layer)})

        if cnf.use_se:
            layers.update({"se": SqueezeExcitation(cnf.input_c,
                                                   cnf.expanded_c)})

        # project
        layers.update({"project_conv": ConvBNActivation(cnf.expanded_c,
                                                        cnf.out_c,
                                                        kernel_size=1,
                                                        norm_layer=norm_layer,
                                                        activation_layer=nn.Identity)})

        self.block = nn.Sequential(layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

        # 只有在使用shortcut连接时才使用dropout层
        if self.use_res_connect and cnf.drop_rate > 0:
            self.dropout = DropPath(cnf.drop_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x) :
        result = self.block(x)
        result = self.dropout(result)
        if self.use_res_connect:
            result += x

        return result


"""

"""


class EfficientNet(nn.Module):
    def __init__(self,
                 width_coefficient: float,
                 depth_coefficient: float,
                 num_classes: int = 1000,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None
                 ):
        super(EfficientNet, self).__init__()

        # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate, repeats
        default_cnf = [[3, 32, 16, 1, 1, True, drop_connect_rate, 1],
                       [3, 16, 24, 6, 2, True, drop_connect_rate, 2],
                       [5, 24, 40, 6, 2, True, drop_connect_rate, 2],
                       [3, 40, 80, 6, 2, True, drop_connect_rate, 3],
                       [5, 80, 112, 6, 1, True, drop_connect_rate, 3],
                       [5, 112, 192, 6, 2, True, drop_connect_rate, 4],
                       [3, 192, 320, 6, 1, True, drop_connect_rate, 1]]

        def round_repeats(repeats):
            """Round number of repeats based on depth multiplier."""
            return int(math.ceil(depth_coefficient * repeats))

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        adjust_channels = partial(InvertedResidualConfig.adjust_channels,
                                  width_coefficient=width_coefficient)

        # build inverted_residual_setting
        bneck_conf = partial(InvertedResidualConfig,
                             width_coefficient=width_coefficient)

        b = 0
        num_blocks = float(sum(round_repeats(i[-1]) for i in default_cnf))
        inverted_residual_setting = []
        for stage, args in enumerate(default_cnf):
            cnf = copy.copy(args)
            for i in range(round_repeats(cnf.pop(-1))):
                if i > 0:
                    # strides equal 1 except first cnf
                    cnf[-3] = 1  # strides
                    cnf[1] = cnf[2]  # input_channel equal output_channel

                cnf[-1] = args[-2] * b / num_blocks  # update dropout ratio
                index = str(stage + 1) + chr(i + 97)  # 1a, 2a, 2b, ...
                inverted_residual_setting.append(bneck_conf(*cnf, index))
                b += 1

        # create layers
        layers = OrderedDict()

        # first conv
        layers.update({"stem_conv": ConvBNActivation(in_planes=3,
                                                     out_planes=adjust_channels(32),
                                                     kernel_size=3,
                                                     stride=2,
                                                     norm_layer=norm_layer)})

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.update({cnf.index: block(cnf, norm_layer)})

        # build top
        last_conv_input_c = inverted_residual_setting[-1].out_c
        last_conv_output_c = adjust_channels(1280)
        layers.update({"top": ConvBNActivation(in_planes=last_conv_input_c,
                                               out_planes=last_conv_output_c,
                                               kernel_size=1,
                                               norm_layer=norm_layer)})

        self.features = nn.Sequential(layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        classifier = []
        if dropout_rate > 0:
            classifier.append(nn.Dropout(p=dropout_rate, inplace=True))
        classifier.append(nn.Linear(last_conv_output_c, num_classes))
        self.classifier = nn.Sequential(*classifier)

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def forward(self, x) :
        return self._forward_impl(x)



def EfficientNetB3(num_classes=1000):
    # input image size 300x300
    return EfficientNet(width_coefficient=1.2,
                        depth_coefficient=1.4,
                        dropout_rate=0.3,
                        num_classes=num_classes)







class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])






class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # self.fundus_branch = EfficientNet.from_pretrained('efficientnet-b3', num_classes=1000)
        self.fundus_branch = EfficientNetB3(num_classes=1000)
        # self.oct_branch = models.resnet34(pretrained=True)
        self.oct_branch = ResNet34()
        self.oct_branch.conv1 = nn.Conv2d(256, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.oct_branch.fc = nn.Identity()  # Remove the fully connected layer

        # 最终的分类数为3
        self.decision_branch = nn.Linear(1512, 3)

    def forward(self, fundus_img, oct_img):
        b1 = self.fundus_branch(fundus_img)
        b2 = self.oct_branch(oct_img)
        b2 = b2.view(b2.size(0), -1)  # Flatten the output of the ResNet branch
        logit = self.decision_branch(torch.cat([b1, b2], 1))
        return logit

def getDataLoader(batchsize,num_workers):
    # 划分训练集和测试集的比例
    val_ratio = 0.20  # 80 / 20
    root_dit = "/data1/detrgroup/cy/20240624/"
    # 训练数据根目录
    trainset_root = root_dit + "Glaucoma_grading/training/multi-modality_images"
    # 标签文件名
    gt_file = root_dit + "Glaucoma_grading/training/glaucoma_grading_training_GT.xlsx"
    # 测试数据根目录
    testset_root = root_dit + "Glaucoma_grading/testing/multi-modality_images"
    # 读取所有训练数据文件名
    filelists = os.listdir(trainset_root)
    # 按照划分比例进行划分
    train_filelists, val_filelists = train_test_split(
        filelists, test_size=val_ratio, random_state=42
    )
    print(
        "Total Nums: {}, train: {}, val: {}".format(
            len(filelists), len(train_filelists), len(val_filelists)
        )
    )
    train_dataset = GAMMA_sub1_dataset(
        dataset_root=trainset_root,
        img_transforms=img_train_transforms,
        oct_transforms=oct_train_transforms,
        filelists=train_filelists,
        label_file=gt_file,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=num_workers,
        batch_size=batchsize,
        shuffle=True,
    )
    val_dataset = GAMMA_sub1_dataset(
        dataset_root=trainset_root,
        filelists=val_filelists,
        label_file=gt_file,
        img_transforms=img_val_transforms,
        oct_transforms=oct_val_transforms,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize,
        num_workers=num_workers

    )
    return train_loader,val_loader


#hyper parameter
num_workers = 4
batchsize = 4
iters = 2000
optimizer_type = "adam"
init_lr = 1e-4


train_loader,val_loader =getDataLoader(batchsize,num_workers)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = Model().to(device)


def val(model, val_dataloader, criterion):
    model.eval()
    avg_loss_list = []
    cache = []
    with torch.no_grad():
        for data in val_dataloader:
            fundus_imgs = (data[0] / 255.0).to(torch.float32).to(device)
            oct_imgs = (data[1] / 255.0).to(torch.float32).to(device)
            labels = data[2].to(torch.int64).to(device)
            fundus_imgs = fundus_imgs.clone().detach().requires_grad_(True)
            oct_imgs = oct_imgs.clone().detach().requires_grad_(True)
            logits = model(fundus_imgs, oct_imgs)

            for p, l in zip(logits.cpu().detach().numpy().argmax(1), labels.cpu().detach().numpy()):
                cache.append([p, l])

            loss = criterion(logits, labels)
            avg_loss_list.append(loss.cpu().detach().numpy().item())

    cache = np.array(cache)
    kappa = cohen_kappa_score(cache[:, 0], cache[:, 1], weights="quadratic")
    avg_loss = np.array(avg_loss_list).mean()

    return avg_loss, kappa


# 训练逻辑
def train(
    model,
    iters,
    train_dataloader,
    val_dataloader,
    optimizer,
    criterion,
    log_interval,
    eval_interval,
):
    iter1 = 0
    model.train()
    avg_loss_list = []
    avg_kappa_list = []
    best_kappa = 0.5
    # breakpoint()
    while iter1 < iters:
        # breakpoint()
        for data in iter(train_dataloader):
            iter1 += 1
            # breakpoint()
            if iter1 > iters:
                break
            fundus_imgs = (data[0] / 255.0).to(torch.float32).to(device)
            oct_imgs = (data[1] / 255.0).to(torch.float32).to(device)
            labels = data[2].to(torch.int64).to(device)
            fundus_imgs = fundus_imgs.clone().detach().requires_grad_(True)
            oct_imgs = oct_imgs.clone().detach().requires_grad_(True)
            logits = model(fundus_imgs, oct_imgs)
            loss = criterion(logits, labels)
            for p, l in zip(logits.cpu().detach().numpy().argmax(1), labels.cpu().detach().numpy()):
                avg_kappa_list.append([p, l])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # model.clear_gradients()
            if loss.cpu().detach().numpy():
                avg_loss_list.append(loss.cpu().detach().numpy().item())

            if iter1 % log_interval == 0:
                avg_loss = np.array(avg_loss_list).mean()
                avg_kappa_list = np.array(avg_kappa_list)
                # 计算Cohen’s kappa分数
                avg_kappa = cohen_kappa_score(
                    avg_kappa_list[:, 0],
                    avg_kappa_list[:, 1],
                    weights="quadratic",
                )
                avg_loss_list = []
                avg_kappa_list = []
                print(
                    "[TRAIN] iter={}/{} avg_loss={:.4f} avg_kappa={:.4f}".format(
                        iter1, iters, avg_loss, avg_kappa
                    )
                )

            if iter1 % eval_interval == 0:
                avg_loss, avg_kappa = val(model, val_dataloader, criterion)
                print(
                    "[EVAL] iter={}/{} avg_loss={:.4f} kappa={:.4f}".format(
                        iter, iters, avg_loss, avg_kappa
                    )
                )
                # 保存精度更好的模型
                if avg_kappa >= best_kappa:
                    best_kappa = avg_kappa
                    if not os.path.exists("best_model_{:.4f}".format(best_kappa)):
                        os.makedirs("best_model_{:.4f}".format(best_kappa))
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            "best_model_{:.4f}".format(best_kappa),
                            "model.pdparams",
                        ),
                    )
                model.train()


import torch.optim as optim
optimizer = optim.Adam( model.parameters(),lr=init_lr)
# ----- 使用交叉熵作为损失函数 ------#
criterion = nn.CrossEntropyLoss()
# ----- 训练模型 -----------------#
train(
    model,
    iters,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    log_interval=10,
    eval_interval=50,
)