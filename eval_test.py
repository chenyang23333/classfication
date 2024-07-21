
from torch.utils.data import DataLoader

from tqdm import tqdm

import os
import numpy as np
import cv2
import pandas as pd
import torch
import torch.nn as nn
import random

from torchvision import models
from efficientnet_pytorch import EfficientNet
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

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fundus_branch = EfficientNet.from_pretrained('efficientnet-b3', num_classes=1000)
        self.oct_branch = models.resnet34(pretrained=True)
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

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# 加载训练权重
best_model_path = "./best_model_0.7834/model.pdparams"  # 注意文件扩展名可能不同
model = Model()
model.load_state_dict(torch.load(best_model_path, map_location=device))  # 添加map_location参数
print("load model success")
model = model.to(device)  # 将模型移动到GPU上
model.eval()

# 定义数据增强操作
img_test_transforms = augment_image
oct_test_transforms = augment_image

root_dit = "/data1/detrgroup/cy/20240624/"
testset_root = root_dit + "Glaucoma_grading/testing/multi-modality_images"

# 创建测试数据集
test_dataset = GAMMA_sub1_dataset(
    dataset_root=testset_root,
    img_transforms=img_test_transforms,
    oct_transforms=oct_test_transforms,
    mode="test",
)

cache = []
# 使用DataLoader可以并行加载数据，提高效率
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

for fundus_img, oct_img, idx in tqdm(test_loader):
    # PyTorch自动处理batch维度，因此不需要手动增加维度
    fundus_img = fundus_img.float().to(device)  / 255.0  # 直接转换为float并归一化
    oct_img = oct_img.float().to(device)  / 255.0

    with torch.no_grad():  # 禁止梯度计算，节省内存
        logits = model(fundus_img, oct_img)

        cache.append([int(idx[0]), logits.argmax(dim=1).item()])  # .item()用于提取单个数值
print(cache)

