from typing import List, Tuple, Optional
from enum import Enum

class TaskPriority(Enum):
    NORMAL = "normal"
    HIGH = "high"

class Task:
    def __init__(self, 
                 task_id: str, 
                 departure: str, 
                 pickup_point: Tuple[int, int],
                 destination: str, 
                 destination_coordinate: Tuple[int, int],
                 priority: TaskPriority, 
                 remaining_time: Optional[int],
                 obstacles: List[Tuple[int, int]],
                 map_size: Tuple[int, int]):
        # Basic information
        self.id: str = task_id  # Task unique identifier
        self.departure: str = departure  # Pickup location name
        self.destination: str = destination  # Destination name
        self.priority: TaskPriority = priority  # Task priority
        self.remaining_time: Optional[int] = remaining_time  # Remaining time: used only for urgent tasks

        self.pickup_point: Tuple[int, int] = pickup_point # Pickup point coordinates (x, y)
        self.destination_coordinate: Tuple[int, int] = destination_coordinate # Destination coordinates (x, y)
        self.obstacles: List[Tuple[int, int]] = obstacles  # Obstacle list
        self.map_size: Tuple[int, int] = map_size  # Map size (width, height)

        # Status informationus information
        self.assigned_agent_id: Optional[str] = None  # Assigned AGV ID
        self.loaded_agent_id: Optional[str] = None
        self.available_timestamp: int = 0  # Task available timestamp
        self.start_timestamp: Optional[int] = None  # Task pickup time
        self.end_timestamp: Optional[int] = None    # Task delivery time
        self.completed: bool = False  # Whether delivered
    
    def is_urgent(self) -> bool:
        return self.priority == TaskPriority.HIGH

    def picked_by(self, agent_id: str, timestamp: int):
        """Mark task as picked up"""
        self.loaded_agent_id = agent_id
        self.start_timestamp = timestamp

    def unload(self, timestamp: int):
        """Mark task as unloaded"""
        self.end_timestamp = timestamp
        self.completed = True

    def is_pending(self) -> bool:
        """Check if task is pending pickup"""
        return self.loaded_agent_id is None
    
    def assign_to(self, agent_id: str):
        """Assign AGV to task"""
        self.assigned_agent_id = agent_id
    
    def is_assigned(self) -> bool:
        """Check if task has been assigned an AGV"""
        return self.assigned_agent_id is not None

    def is_delivering(self) -> bool:
        """Check if task is being delivered"""
        return self.loaded_agent_id is not None and not self.completed
    
    def is_completed(self) -> bool:
        """Check if task is completed"""
        return self.completed
    
    def get_induction_station_id(self) -> str:
        """Get induction station ID corresponding to task"""
        return self.departure
    
    def get_destination_name(self) -> str:
        """Get task destination name"""
        return self.destination

    def get_destination_coordinate(self) -> Tuple[int, int]:
        """Get destination slot coordinates"""
        return self.destination_coordinate

    def get_pickup_points(self) -> Tuple[int, int]:
        """Get pickup point coordinates from induction station"""
        return self.pickup_point

    def get_delivery_points(self) -> List[Tuple[int, int]]:
        """Get all delivery point coordinates of destination slot"""
        dest_x, dest_y = self.get_destination_coordinate()
        
        # Each slot has 4 adjacent delivery points (up, down, left, right)
        delivery_points = [
            (dest_x, dest_y + 1),  # Up
            (dest_x, dest_y - 1),  # Down
            (dest_x - 1, dest_y),  # Left
            (dest_x + 1, dest_y)   # Right
        ]
        
        # Filter out delivery points outside map range
        valid_points = []
        for point in delivery_points:
            x, y = point
            if 1 <= x <= self.map_size[0] and 1 <= y <= self.map_size[1] and point not in self.obstacles:
                valid_points.append(point)
        
        return valid_points

