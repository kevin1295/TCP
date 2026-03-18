import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
from typing import Tuple, Optional
import random


class DynamicGaussianBlur:
    """动态高斯模糊，对应原imgaug的GaussianBlur"""
    def __init__(self, sigma_range: Tuple[float, float], p: float = 0.5):
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.uniform(*self.sigma_range)
            return F.gaussian_blur(img, kernel_size=5, sigma=sigma)
        return img


class AdditiveGaussianNoise:
    """加性高斯噪声，对应原imgaug的AdditiveGaussianNoise"""
    def __init__(self, scale_range: Tuple[float, float], per_channel: float = 1.0, p: float = 0.5):
        self.scale_range = scale_range
        self.per_channel = per_channel
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            # 将PIL图像转为张量 (0-1范围)
            img_tensor = F.pil_to_tensor(img).float() / 255.0
            c, h, w = img_tensor.shape
            
            # 生成噪声
            scale = random.uniform(*self.scale_range)
            noise = torch.randn_like(img_tensor) * scale
            
            # 按通道概率应用噪声
            if self.per_channel < 1.0:
                channel_mask = torch.rand(c) < self.per_channel
                noise = noise * channel_mask.view(c, 1, 1)
            
            # 应用噪声并裁剪到有效范围
            img_tensor = img_tensor + noise
            img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
            
            # 转回PIL图像
            return F.to_pil_image(img_tensor * 255.0)
        return img


class CoarseDropout:
    """粗dropout，对应原imgaug的CoarseDropout"""
    def __init__(self, dropout_range: Tuple[float, float], size_percent_range: Tuple[float, float], 
                 per_channel: float = 1.0, p: float = 0.5):
        self.dropout_range = dropout_range
        self.size_percent_range = size_percent_range
        self.per_channel = per_channel
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img_tensor = F.pil_to_tensor(img).float() / 255.0
            c, h, w = img_tensor.shape
            
            # 计算dropout参数
            dropout_prob = random.uniform(*self.dropout_range)
            size_percent = random.uniform(*self.size_percent_range)
            patch_size = (int(h * size_percent), int(w * size_percent))
            
            # 生成mask
            mask = torch.ones_like(img_tensor)
            num_patches = int(dropout_prob * h * w / (patch_size[0] * patch_size[1]))
            
            for _ in range(num_patches):
                # 随机选择patch位置
                y = random.randint(0, h - patch_size[0])
                x = random.randint(0, w - patch_size[1])
                
                # 按通道应用dropout
                for ch in range(c):
                    if random.random() < self.per_channel:
                        mask[ch, y:y+patch_size[0], x:x+patch_size[1]] = 0.0
            
            # 应用mask
            img_tensor = img_tensor * mask
            img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
            
            return F.to_pil_image(img_tensor * 255.0)
        return img


class Dropout:
    """普通dropout，对应原imgaug的Dropout"""
    def __init__(self, dropout_range: Tuple[float, float], per_channel: float = 1.0, p: float = 0.5):
        self.dropout_range = dropout_range
        self.per_channel = per_channel
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img_tensor = F.pil_to_tensor(img).float() / 255.0
            c, h, w = img_tensor.shape
            
            # 生成dropout mask
            dropout_prob = random.uniform(*self.dropout_range)
            mask = torch.ones_like(img_tensor)
            
            for ch in range(c):
                if random.random() < self.per_channel:
                    ch_mask = torch.rand(h, w) > dropout_prob
                    mask[ch] = ch_mask.float()
            
            # 应用mask
            img_tensor = img_tensor * mask
            img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
            
            return F.to_pil_image(img_tensor * 255.0)
        return img


class AddBrightness:
    """亮度添加，对应原imgaug的Add"""
    def __init__(self, add_range: Tuple[float, float], per_channel: float = 1.0, p: float = 0.5):
        self.add_range = add_range
        self.per_channel = per_channel
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            # Add操作的范围是基于像素值(0-255)的，需要转换
            add_value = random.uniform(*self.add_range) / 255.0
            img_tensor = F.pil_to_tensor(img).float() / 255.0
            c, h, w = img_tensor.shape
            
            # 按通道应用亮度调整
            for ch in range(c):
                if random.random() < self.per_channel:
                    img_tensor[ch] = torch.clamp(img_tensor[ch] + add_value, 0.0, 1.0)
            
            return F.to_pil_image(img_tensor * 255.0)
        return img


class MultiplyBrightness:
    """亮度乘法，对应原imgaug的Multiply"""
    def __init__(self, multiply_range: Tuple[float, float], per_channel: float = 1.0, p: float = 0.5):
        self.multiply_range = multiply_range
        self.per_channel = per_channel
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            multiply_value = random.uniform(*self.multiply_range)
            img_tensor = F.pil_to_tensor(img).float() / 255.0
            c, h, w = img_tensor.shape
            
            # 按通道应用亮度调整
            for ch in range(c):
                if random.random() < self.per_channel:
                    img_tensor[ch] = torch.clamp(img_tensor[ch] * multiply_value, 0.0, 1.0)
            
            return F.to_pil_image(img_tensor * 255.0)
        return img


class LinearContrast:
    """线性对比度调整，对应原imgaug的LinearContrast"""
    def __init__(self, contrast_range: Tuple[float, float], per_channel: float = 1.0, p: float = 0.5):
        self.contrast_range = contrast_range
        self.per_channel = per_channel
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            contrast_factor = random.uniform(*self.contrast_range)
            img_tensor = F.pil_to_tensor(img).float() / 255.0
            c, h, w = img_tensor.shape
            
            # 按通道应用对比度调整
            for ch in range(c):
                if random.random() < self.per_channel:
                    # 对比度调整公式: (x - 0.5) * factor + 0.5
                    img_tensor[ch] = (img_tensor[ch] - 0.5) * contrast_factor + 0.5
                    img_tensor[ch] = torch.clamp(img_tensor[ch], 0.0, 1.0)
            
            return F.to_pil_image(img_tensor * 255.0)
        return img


class RandomGrayscale:
    """随机灰度化，增强版本以匹配imgaug的Grayscale"""
    def __init__(self, grayscale_range: Tuple[float, float], p: float = 0.5):
        self.grayscale_range = grayscale_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            # 随机生成灰度化程度
            alpha = random.uniform(*self.grayscale_range)
            if alpha == 0:
                return img
            # 转为灰度图
            gray_img = F.rgb_to_grayscale(img, num_output_channels=3)
            # 混合原图和灰度图
            return F.blend(img, gray_img, alpha=alpha)
        return img


class RandomOrder:
    """随机顺序应用变换，对应imgaug的random_order"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        # 随机打乱变换顺序
        shuffled_transforms = random.sample(self.transforms, len(self.transforms))
        for t in shuffled_transforms:
            img = t(img)
        return img


def hard(image_iteration):
    """
    原hard函数的torchvision实现
    """
    iteration = image_iteration / 32
    frequency_factor = min(0.05 + float(iteration)/200000.0, 1.0)
    color_factor = min(float(iteration)/1000000.0, 1.0)
    dropout_factor = 0.198667 + (0.03856658 - 0.198667) / (1 + (iteration / 196416.6) ** 1.863486)
    blur_factor = min(0.5 + (0.5*iteration/100000.0), 1.0)
    add_factor = 10 + 10*iteration/100000.0
    multiply_factor_pos = 1 + (2.5*iteration/200000.0)
    multiply_factor_neg = 1 - (0.91 * iteration / 500000.0)
    contrast_factor_pos = 1 + (0.5*iteration/500000.0)
    contrast_factor_neg = 1 - (0.5 * iteration / 500000.0)

    # 构建变换列表
    transforms = [
        # 高斯模糊
        DynamicGaussianBlur(
            sigma_range=(0, blur_factor),
            p=frequency_factor
        ),
        # 加性高斯噪声
        AdditiveGaussianNoise(
            scale_range=(0.0, dropout_factor),
            per_channel=color_factor,
            p=frequency_factor
        ),
        # CoarseDropout
        CoarseDropout(
            dropout_range=(0.0, dropout_factor),
            size_percent_range=(0.08, 0.2),
            per_channel=color_factor,
            p=frequency_factor
        ),
        # Dropout
        Dropout(
            dropout_range=(0.0, dropout_factor),
            per_channel=color_factor,
            p=frequency_factor
        ),
        # 亮度添加
        AddBrightness(
            add_range=(-add_factor, add_factor),
            per_channel=color_factor,
            p=frequency_factor
        ),
        # 亮度乘法
        MultiplyBrightness(
            multiply_range=(multiply_factor_neg, multiply_factor_pos),
            per_channel=color_factor,
            p=frequency_factor
        ),
        # 对比度调整
        LinearContrast(
            contrast_range=(contrast_factor_neg, contrast_factor_pos),
            per_channel=color_factor,
            p=frequency_factor
        ),
        # 灰度化
        RandomGrayscale(
            grayscale_range=(0.0, 1),
            p=frequency_factor
        )
    ]

    # 随机顺序应用变换
    augmenter = RandomOrder(transforms)
    
    return augmenter


def hard_1(image_iteration):
    """
    原hard_1函数的torchvision实现（仅CoarseDropout的size_percent不同）
    """
    iteration = image_iteration / 32
    frequency_factor = min(0.05 + float(iteration)/200000.0, 1.0)
    color_factor = min(float(iteration)/1000000.0, 1.0)
    dropout_factor = 0.198667 + (0.03856658 - 0.198667) / (1 + (iteration / 196416.6) ** 1.863486)
    blur_factor = min(0.5 + (0.5*iteration/100000.0), 1.0)
    add_factor = 10 + 10*iteration/100000.0
    multiply_factor_pos = 1 + (2.5*iteration/200000.0)
    multiply_factor_neg = 1 - (0.91 * iteration / 500000.0)
    contrast_factor_pos = 1 + (0.5*iteration/500000.0)
    contrast_factor_neg = 1 - (0.5 * iteration / 500000.0)

    # 构建变换列表（仅CoarseDropout的size_percent_range不同）
    transforms = [
        DynamicGaussianBlur(
            sigma_range=(0, blur_factor),
            p=frequency_factor
        ),
        AdditiveGaussianNoise(
            scale_range=(0.0, dropout_factor),
            per_channel=color_factor,
            p=frequency_factor
        ),
        CoarseDropout(
            dropout_range=(0.0, dropout_factor),
            size_percent_range=(0.02, 0.2),  # hard_1的区别
            per_channel=color_factor,
            p=frequency_factor
        ),
        Dropout(
            dropout_range=(0.0, dropout_factor),
            per_channel=color_factor,
            p=frequency_factor
        ),
        AddBrightness(
            add_range=(-add_factor, add_factor),
            per_channel=color_factor,
            p=frequency_factor
        ),
        MultiplyBrightness(
            multiply_range=(multiply_factor_neg, multiply_factor_pos),
            per_channel=color_factor,
            p=frequency_factor
        ),
        LinearContrast(
            contrast_range=(contrast_factor_neg, contrast_factor_pos),
            per_channel=color_factor,
            p=frequency_factor
        ),
        RandomGrayscale(
            grayscale_range=(0.0, 1),
            p=frequency_factor
        )
    ]

    augmenter = RandomOrder(transforms)
    
    return augmenter


# 使用示例
if __name__ == "__main__":
    from PIL import Image
    import os

    # 加载示例图像
    img_path = "example.jpg"  # 替换为你的图像路径
    if os.path.exists(img_path):
        img = Image.open(img_path).convert("RGB")
        
        # 获取增强器（假设迭代次数为10000）
        augmenter = hard(10000)
        
        # 应用增强
        augmented_img = augmenter(img)
        
        # 保存结果
        augmented_img.save("augmented_example.jpg")
        print("增强后的图像已保存为 augmented_example.jpg")
    else:
        print(f"请提供有效的图像路径，当前路径 {img_path} 不存在")