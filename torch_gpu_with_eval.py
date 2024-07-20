import os
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score


import torch
import torch.nn as nn


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

        # 对彩色眼底图片进行数据增强
        # if self.img_transforms is not None:
        #     from PIL import Image
        #     if isinstance(fundus_img, np.ndarray):
        #         fundus_img = Image.fromarray(fundus_img)
        #     fundus_img = self.img_transforms(fundus_img)

        # 对3D OCT图片进行数据增强
        # if self.oct_transforms is not None:
        #     breakpoint()
        #     # from PIL import Image
        #     # if isinstance(oct_img, np.ndarray):
        #     #     oct_img = Image.fromarray(oct_img)
        #     oct_img = self.oct_transforms(oct_img)

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

# 数据增强操作
img_train_transforms = trans.Compose(
    [
        # 按比例随机裁剪原图后放缩到对应大小
        trans.RandomResizedCrop(
            image_size, scale=(0.90, 1.1), ratio=(0.90, 1.1)
        ),
        # 随机水平翻转
        trans.RandomHorizontalFlip(),
        # 随机垂直翻转
        trans.RandomVerticalFlip(),
        # 随机旋转
        trans.RandomRotation(30),
    ]
)

oct_train_transforms = trans.Compose(
    [
        # 中心裁剪到对应大小
        trans.CenterCrop(oct_img_size),
        # 随机水平翻转
        trans.RandomHorizontalFlip(),
        # 随机垂直翻转
        trans.RandomVerticalFlip(),
    ]
)

img_val_transforms = trans.Compose(
    [
        # 将图片放缩到固定大小
        trans.Resize(image_size)
    ]
)

oct_val_transforms = trans.Compose(
    [
        # 中心裁剪到固定大小
        trans.CenterCrop(oct_img_size)
    ]
)


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
batchsize = 1
iters = 2000
optimizer_type = "adam"
init_lr = 1e-3


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
    log_interval=50,
    eval_interval=100,
)