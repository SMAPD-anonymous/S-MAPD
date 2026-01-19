import csv
import os
from config_manager import ConfigManager

class DataLoader:
    """Data loader for reading task and map data from CSV files"""

    @staticmethod
    def get_abs_filepath(filename):
        """Get absolute path of file"""
        return os.path.join(os.getcwd(), filename)

    @staticmethod
    def get_task_path():
        """Get absolute path of task file"""
        config = ConfigManager.load_config()
        if config is None:
            raise ValueError("Unable to load config file, please ensure settings.json exists")
        file_names = ConfigManager.create_file_names(config)
        return DataLoader.get_abs_filepath(os.path.join("data", file_names["task_filename"]))
    
    @staticmethod
    def get_map_path():
        """Get absolute path of map file"""
        config = ConfigManager.load_config()
        if config is None:
            raise ValueError("Unable to load config file, please ensure settings.json exists")
        file_names = ConfigManager.create_file_names(config)
        return DataLoader.get_abs_filepath(os.path.join("data", file_names["map_filename"]))
    
    @staticmethod
    def get_map_type():
        """Get map type"""
        config = ConfigManager.load_config()
        if config is None:
            raise ValueError("Unable to load config file, please ensure settings.json exists")
        return config['scene_size'] + "_" + config['map_distribution']
    
    @staticmethod
    def get_trajectory_path():
        """Get absolute path of AGV trajectory file"""
        config = ConfigManager.load_config()
        if config is None:
            raise ValueError("Unable to load config file, please ensure settings.json exists")
        file_names = ConfigManager.create_file_names(config)
        return DataLoader.get_abs_filepath(os.path.join("data", file_names["trajectory_filename"]))
    
    @staticmethod
    def get_tasks():
        """Read task data from CSV file"""
        tasks = []
        task_path = DataLoader.get_task_path()
        try:
            print(f"Attempting to read task file: {task_path}")
            with open(task_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    tasks.append({
                        "task_id": row["task_id"],
                        "start_point": row["start_point"].strip(),
                        "end_point": row["end_point"].strip(),
                        "priority": row["priority"],
                        "remaining_time": int(row["remaining_time"]) if row["remaining_time"] not in [None, "", "None"] else None
                    })
        except FileNotFoundError:
            print(f"Task file not found, please ensure {task_path} exists.")
            print("Please run DataGenerator.py to generate data files")
        except Exception as e:
            print(f"Error reading task file: {str(e)}")
        return tasks

    @staticmethod
    def get_map_elements():
        """Read map data from CSV file"""
        start_points, end_points, agv_list, obstacle_list, map_size = {}, {}, [], [], None
        map_path = DataLoader.get_map_path()
        try:
            print(f"Attempting to read map file: {map_path}")
            with open(map_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    t, name = row["type"].strip(), row["name"].strip()
                    x, y = int(row["x"]), int(row["y"])
                    if t == "start_point":
                        start_points[name] = (x, y)
                    elif t == "end_point":
                        end_points[name] = (x, y)
                    elif t == "agv":
                        agv_list.append({
                            "id": name,
                            "pos": (x, y),
                            "pitch": int(row["pitch"])
                        })
                    elif t == "obstacle":
                        obstacle_list.append((x, y))
                    elif t == "map_size":
                        map_size = (x, y)
        except FileNotFoundError:
            print(f"Map file not found, please ensure {map_path} exists.")
            print("Please run DataGenerator.py to generate data files")
        except Exception as e:
            print(f"Error reading map file: {str(e)}")
        return start_points, end_points, agv_list, obstacle_list, map_size