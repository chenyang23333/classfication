import torch
from torch.utils.data import DataLoader
from torchvision import transforms as trans
from tqdm import tqdm
import numpy as np
from torch_gpu_with_eval import Model  # 替换为你的模型文件名
from torch_gpu_with_eval import GAMMA_sub1_dataset,augment_image # 替换为你的数据集文件名

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# 加载训练权重
best_model_path = "./best_model_0.7834/model.pdparams"  # 注意文件扩展名可能不同
model = Model()
model.load_state_dict(torch.load(best_model_path, map_location=device))  # 添加map_location参数
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
        cache.append([idx.item(), logits.argmax(dim=1).item()])  # .item()用于提取单个数值

