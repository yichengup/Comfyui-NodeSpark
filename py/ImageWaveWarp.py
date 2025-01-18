import torch
import numpy as np
import cv2

class ImageWaveWarp:
    """图像波浪扭曲节点"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "wave_frequency": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "wave_direction": (["horizontal", "vertical", "radial"],),
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "wave_warp"
    CATEGORY = "NodeSpark/image"
    
    @classmethod
    def IS_CHANGED(cls):
        return True
        
    @classmethod
    def VALIDATE_INPUTS(cls, *args, **kwargs):
        return True

    def __init__(self):
        self.class_type = "ImageWaveWarp"

    def wave_warp(self, image, strength, wave_frequency, wave_direction, center_x, center_y):
        # 转换图像格式
        img = (image.cpu().numpy()[0] * 255).astype(np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
        height, width = img.shape[:2]
        center_x = int(width * center_x)
        center_y = int(height * center_y)
        
        # 创建网格
        y, x = np.indices((height, width))
        
        # 根据波浪方向计算偏移
        if wave_direction == "horizontal":
            # 水平波浪
            phase = y / height * 2 * np.pi * wave_frequency
            x_offset = np.sin(phase) * strength * width / 20
            y_offset = np.zeros_like(x_offset)
        elif wave_direction == "vertical":
            # 垂直波浪
            phase = x / width * 2 * np.pi * wave_frequency
            x_offset = np.zeros_like(x)
            y_offset = np.sin(phase) * strength * height / 20
        else:  # radial
            # 径向波浪
            dx = x - center_x
            dy = y - center_y
            r = np.sqrt(dx**2 + dy**2)
            phase = r / (width/2) * 2 * np.pi * wave_frequency
            angle = np.arctan2(dy, dx)
            magnitude = np.sin(phase) * strength * width / 20
            x_offset = magnitude * np.cos(angle)
            y_offset = magnitude * np.sin(angle)
        
        # 应用偏移
        x_new = x + x_offset
        y_new = y + y_offset
        
        # 确保坐标在有效范围内
        x_new = np.clip(x_new, 0, width-1)
        y_new = np.clip(y_new, 0, height-1)
        
        # 应用变形并转换回tensor格式
        result = cv2.remap(img, x_new.astype(np.float32), y_new.astype(np.float32), cv2.INTER_LINEAR)
        result = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)
        return (result,)

NODE_CLASS_MAPPINGS = {
    "ImageWaveWarp": ImageWaveWarp
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageWaveWarp": "Image Wave Warp"
} 