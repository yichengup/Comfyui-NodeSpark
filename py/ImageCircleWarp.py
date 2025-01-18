import torch
import numpy as np
import cv2

class ImageCircleWarp:
    """图像圆形扭曲节点"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "radius": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "warp_image"
    CATEGORY = "NodeSpark/image"
    
    @classmethod
    def IS_CHANGED(cls):
        return True
        
    @classmethod
    def VALIDATE_INPUTS(cls, *args, **kwargs):
        return True

    def __init__(self):
        self.class_type = "ImageCircleWarp"

    def ellipse_warp(self, img, strength, radius, center_x, center_y):
        height, width = img.shape[:2]
        center_x = int(width * center_x)
        center_y = int(height * center_y)
        
        # 创建网格
        y, x = np.indices((height, width))
        
        # 计算到中心点的归一化距离
        dx = (x - center_x) / (width/2)
        dy = (y - center_y) / (height/2)
        r = np.sqrt(dx**2 + dy**2)
        
        # 计算影响因子，基于radius参数
        influence = np.clip(1.0 - r / radius, 0, 1)
        
        # 应用平滑过渡
        influence = influence * influence * (3 - 2 * influence)
        
        # 计算变形强度，只在radius范围内应用
        scale = 1.0 + strength * influence
        
        # 应用变形
        x_new = center_x + (x - center_x) * scale
        y_new = center_y + (y - center_y) * scale
        
        # 确保坐标在有效范围内
        x_new = np.clip(x_new, 0, width-1)
        y_new = np.clip(y_new, 0, height-1)
        
        return cv2.remap(img, x_new.astype(np.float32), y_new.astype(np.float32), cv2.INTER_LINEAR)

    def warp_image(self, image, strength, radius, center_x, center_y):
        # 转换图像格式
        img = (image.cpu().numpy()[0] * 255).astype(np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # 应用椭圆变形
        result = self.ellipse_warp(img, strength, radius, center_x, center_y)
        
        # 转换回 tensor 格式
        result = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)
        return (result,)

NODE_CLASS_MAPPINGS = {
    "ImageCircleWarp": ImageCircleWarp
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCircleWarp": "Image Circle Warp"
} 