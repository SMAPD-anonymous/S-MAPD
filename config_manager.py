import json
import os
from enum import Enum

class SceneSize(Enum):
    SMALL = "small"
    LARGE = "large"

class MapDistributionType(Enum):
    SORTING_CENTER = "sorting_center"
    LOGISTICS_WAREHOUSE = "logistics_warehouse"
    WORKSHOP = "workshop"

class TaskDistributionType(Enum):
    UNIFORM = "uniform"
    RANDOM = "random"
    SKEWED = "skewed"
    CUSTOM = "custom"

class ConfigManager:
    """Configuration manager for DataLoader to read config"""
    
    CONFIG_FILE = "settings.json"
    
    @staticmethod
    def load_config():
        """Load configuration file"""
        if not os.path.exists(ConfigManager.CONFIG_FILE):
            # If config file doesn't exist, create default config
            default_config = {
                "scene_size": "small",
                "map_distribution": "sorting_center",
                "num_agvs": 10,
                "num_tasks": 500,
                "task_distribution": "uniform"
            }
            with open(ConfigManager.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            print(f"Created default config file: {ConfigManager.CONFIG_FILE}")
            return default_config
        
        try:
            with open(ConfigManager.CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Error reading config file: {str(e)}")
            return None
    
    @staticmethod
    def create_file_names(config):
        """Create filenames based on config"""
        
        map_filename = f"map_{config['scene_size']}_{config['map_distribution']}_AGV{config['num_agvs']}.csv"
        
        task_filename = f"task_{config['scene_size']}_{config['map_distribution']}_{config['task_distribution']}_{config['num_tasks']}.csv"
        
        trajectory_filename = f"agv_trajectory_{config['scene_size']}_{config['map_distribution']}_AGV{config['num_agvs']}_{config['task_distribution']}_{config['num_tasks']}.csv"
        
        return {
            "map_filename": map_filename,
            "task_filename": task_filename,
            "trajectory_filename": trajectory_filename
        }