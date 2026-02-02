import os
import yaml

class Config:
    def __init__(self, config_path="D:/python_project/wire_project/config/config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        
    def load_config(self):
        """加载YAML配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 将列表转换为numpy数组
        import numpy as np
        
        # 转换相机内参
        config['camera']['K'] = np.array(config['camera']['K'], dtype=np.float64)
        
        # 转换畸变系数
        config['camera']['dist_coeffs'] = np.array(config['camera']['dist_coeffs'], dtype=np.float64)
        
        # 转换外参
        config['camera']['rotation_vector'] = np.array(config['camera']['rotation_vector'], dtype=np.float64)
        config['camera']['translation_vector'] = np.array(config['camera']['translation_vector'], dtype=np.float64)
        
        # 转换颜色参数
        config['segmentation']['outline_color'] = tuple(config['segmentation']['outline_color'])
        config['segmentation']['fill_color'] = tuple(config['segmentation']['fill_color'])
        config['point_cloud']['point_color'] = tuple(config['point_cloud']['point_color'])
        config['point_cloud']['history_point_color'] = tuple(config['point_cloud']['history_point_color'])
        
        return config
    
    def get(self, key, default=None):
        """获取配置值，支持点分路径"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def update(self, updates):
        """更新配置"""
        for key, value in updates.items():
            keys = key.split('.')
            config_level = self.config
            for k in keys[:-1]:
                if k not in config_level:
                    config_level[k] = {}
                config_level = config_level[k]
            config_level[keys[-1]] = value