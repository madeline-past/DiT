import torch
import torch.nn as nn
import random
import math

# 训练的数据，应该形状大小完全一致
def generate_data(seed=None, height=400, width=480):
    if seed:
        random.seed(seed)
        torch.manual_seed(seed)

    # 添加随机性的参数
    omiga = 2 * math.pi / (height/10) * random.uniform(0.7, 1.3)  # ±30%波动
    # amplitude = 1 * random.uniform(0.1, 1)                        # 0.1到1
    amplitude = 0.2
    phi = random.uniform(0, 2 * math.pi)                            # 0到2π随机取值
    pivot = random.randint(0, width-1)

    # 时间维上按正弦规律变化的相位大小
    key = torch.sin(torch.arange(0, height).float() * omiga + phi) * amplitude
    key = key.unsqueeze(1)

    # 距离维上的衰减
    margin = 100
    tanh = nn.Tanh()
    weight_1 = tanh(torch.linspace(0, 2, steps=margin+1))
    weight_2 = torch.flip(weight_1[:-1], dims=[0])
    weight = torch.cat([weight_1, weight_2], dim=0).unsqueeze(0)
    feat = key * weight    # (height, 2*margin +1)

    # 处理边界问题
    if pivot + margin > width -1:
        feat = feat[:, :width -1 - (pivot - margin) +1]
        left_pad = pivot - margin
        right_pad = 0
    elif pivot - margin < 0:
        feat = feat[:, 0 - (pivot - margin):]
        left_pad = 0
        right_pad = (width -1) - (pivot + margin)
    else:
        left_pad = pivot - margin
        right_pad = (width -1) - (pivot + margin)

    # 合成带底噪的数据
    feat_pad = nn.functional.pad(feat, (left_pad, right_pad), mode='constant', value=0)     # (height, width)
    noise = torch.randn(height, width)
    assert noise.shape == feat_pad.shape
    raw_data = noise + feat_pad
    # 这里feat_pad相当于没有任何噪音的ground_truth，raw_data是生图过程中提供的图片条件
    # return feat_pad, raw_data
    return feat_pad.unsqueeze(0), raw_data.unsqueeze(0)




# if '__name__' == '__main__':
#     import matplotlib.pyplot as plt
#     array = generate_data().numpy()
#     # plt.figure(figsize=(5, 5))  # 设置图像尺寸
#     plt.imshow(array)  # 使用Matplotlib显示NumPy数组
#     plt.xlabel('distance')  # 添加X轴标签
#     plt.ylabel('time')  # 添加Y轴标签
#     plt.title('raw_data')  # 添加标题
#     plt.show()  # 显示图像
