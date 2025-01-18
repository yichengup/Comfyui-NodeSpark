import torch
import numpy as np
import cv2
from scipy.interpolate import CubicSpline

class ImageStretch:
    """图像拉伸变形节点"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1}),
                "direction": (["horizontal", "vertical"],),
                "position": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.9, "step": 0.01}),
                "stretch_width": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.3, "step": 0.01}),
                "transition": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 0.1, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "funny_mirror"
    CATEGORY = "NodeSpark/image"
    
    @classmethod
    def IS_CHANGED(cls):
        return True
        
    @classmethod
    def VALIDATE_INPUTS(cls, *args, **kwargs):
        return True

    def __init__(self):
        self.class_type = "ImageStretch"

    def create_control_points(self, size, center_pos, stretch_width, transition, strength):
        """创建样条插值的控制点"""
        # 计算关键区域的边界
        half_stretch = stretch_width * size / 2
        stretch_start = max(center_pos - half_stretch, half_stretch)  # 确保不会太靠近边界
        stretch_end = min(center_pos + half_stretch, size - half_stretch)  # 确保不会超出边界
        trans_pixels = min(transition * size, half_stretch)  # 限制过渡区域大小
        
        # 创建更多的控制点以实现更平滑的过渡
        x = np.array([
            0,                                     # 起始点（无变形）
            max(0.1 * size, stretch_start - trans_pixels * 2),  # 远过渡区开始
            max(0.1 * size, stretch_start - trans_pixels),      # 近过渡区开始
            stretch_start,                         # 拉伸区开始
            center_pos,                           # 中心点
            stretch_end,                          # 拉伸区结束
            min(size - 0.1 * size, stretch_end + trans_pixels),      # 近过渡区结束
            min(size - 0.1 * size, stretch_end + trans_pixels * 2),  # 远过渡区结束
            size                                  # 终止点（无变形）
        ])
        
        # 确保x坐标严格递增
        x = np.sort(x)
        eps = 1e-6 * size
        x[1:] = np.maximum(x[1:], x[:-1] + eps)
        
        # 归一化x坐标到[0,1]区间
        x = x / size
        
        # 计算变形量，使用正弦函数实现平滑过渡
        y = np.zeros_like(x)
        
        # 拉伸区域使用固定变形量
        center_region = slice(3, 6)  # 拉伸区域的索引范围
        y[center_region] = (strength - 1) * half_stretch
        
        # 过渡区域使用正弦函数实现平滑过渡
        left_transition = slice(1, 3)   # 左过渡区域
        right_transition = slice(6, 8)  # 右过渡区域
        
        # 左侧过渡
        t_left = np.linspace(0, np.pi/2, len(y[left_transition]))
        y[left_transition] = (strength - 1) * half_stretch * np.sin(t_left)
        
        # 右侧过渡
        t_right = np.linspace(np.pi/2, np.pi, len(y[right_transition]))
        y[right_transition] = (strength - 1) * half_stretch * np.cos(t_right)
        
        # 确保边界点无变形
        y[0] = 0   # 起始点
        y[-1] = 0  # 终止点
        
        return x, y

    def create_spline_mapping(self, size, center_pos, stretch_width, transition, strength):
        """创建基于样条的变形映射"""
        # 创建控制点
        x, y = self.create_control_points(size, center_pos, stretch_width, transition, strength)
        
        # 创建三次样条插值器
        spline = CubicSpline(x, y, bc_type='natural')
        
        # 生成所有位置的映射
        positions = np.linspace(0, 1, size)
        deformations = spline(positions)
        
        # 计算新的坐标映射
        new_positions = np.arange(size, dtype=np.float32)
        new_positions += deformations
        
        return new_positions

    def funny_mirror(self, image, strength, direction, position, stretch_width, transition):
        # 转换图像格式
        img = (image.cpu().numpy()[0] * 255).astype(np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
        height, width = img.shape[:2]
        
        # 创建坐标网格
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # 根据方向应用变形
        if direction == "horizontal":
            center_pos = int(position * height)
            # 创建水平方向的变形映射
            y_new = self.create_spline_mapping(height, center_pos, stretch_width, transition, strength)
            # 应用变形
            y = y_new[y]
            x_new = x
        else:
            center_pos = int(position * width)
            # 创建垂直方向的变形映射
            x_new = self.create_spline_mapping(width, center_pos, stretch_width, transition, strength)
            # 应用变形
            x = x_new[x]
            y_new = y
        
        # 确保坐标在有效范围内
        x = np.clip(x, 0, width-1)
        y = np.clip(y, 0, height-1)
        
        # 应用变形并转换回tensor格式
        result = cv2.remap(img, x.astype(np.float32), 
                          y.astype(np.float32), 
                          cv2.INTER_CUBIC, 
                          borderMode=cv2.BORDER_REFLECT)
        result = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)
        
        return (result,)

NODE_CLASS_MAPPINGS = {
    "ImageStretch": ImageStretch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageStretch": "Image Stretch"
} 