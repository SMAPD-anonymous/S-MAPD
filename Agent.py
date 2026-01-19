from typing import List, Dict, Tuple, Optional
from enum import Enum
from Task import Task

class AgentStatus(Enum):
    IDLE = "idle"           # Idle
    MOVING_TO_PICKUP = "moving_to_pickup"       # Moving to pickup point
    MOVING_TO_DELIVERY = "moving_to_delivery"   # Moving to delivery point
    MOVING_TO_IDLE = "moving_to_idle"         # Moving to idle point

class AgentActions(Enum):
    MOVE = "move"       # Move
    TURN = "turn"       # Turn
    LOAD = "load"     # Loading
    UNLOAD = "unload" # Unloading
    WAIT = "wait"     # Wait

class Agent:
    def __init__(self, agent_id: str, start_x: int, start_y: int, start_pitch: int, map_size: Tuple[int, int], map_path_cost: Dict[Tuple[int, int], Dict[Tuple[int, int], int]]):
        self.id: str = agent_id  # AGV unique identifier, i.e., AGV name
        self.x: int = start_x
        self.y: int = start_y
        self.pitch: int = start_pitch  # Angle: 0-X positive, 90-Y positive, 180-X negative, 270-Y negative
        self.map_size: Tuple[int, int] = map_size  # Map size (width, height)
        self.map_path_cost = map_path_cost

        # State relatede related
        self.status = AgentStatus.IDLE
        self.action = AgentActions.WAIT
        self.current_timestamp: int = 0  # Current timestep
        self.loaded: bool = False  # Whether carrying cargo
        self.current_task: Optional[Task] = None  # Current task
        self.current_destination: Optional[Tuple[int, int, Optional[int]]] = None  # Target point and orientation (x, y, pitch)
        self.path: List[Tuple[int, int, int]] = [] # Current path point list [(x, y, pitch), ...]
        
        # Trajectory recording
        self.trajectory: List[Dict] = []

    def get_position(self) -> Tuple[int, int]:
        """Get AGV current position coordinates"""
        return (self.x, self.y)
    
    def get_pitch(self) -> int:
        """Get AGV current orientation angle"""
        return self.pitch
    
    def get_next_position(self) -> Tuple[int, int]:
        """Get AGV next position coordinates"""
        if self.path and len(self.path) > 0:
            next_pos = self.path[0]
            return (next_pos[0], next_pos[1])
        else:
            return self.get_position()
        
    def get_action(self) -> AgentActions:
        """Get AGV current action"""
        return self.action
        
    def generate_next_action(self):
        """Generate AGV next action"""
        if self.status == AgentStatus.IDLE:
            self.action = AgentActions.WAIT
        elif len(self.path) == 0:
            if self.status == AgentStatus.MOVING_TO_PICKUP:
                self.action = AgentActions.LOAD
            elif self.status == AgentStatus.MOVING_TO_DELIVERY:
                self.action = AgentActions.UNLOAD
            elif self.status == AgentStatus.MOVING_TO_IDLE:
                self.action = AgentActions.WAIT
        else:
            next_x, next_y, next_pitch = self.path[0]
            if self.pitch != next_pitch: # Same position but different orientation, perform turn
                self.action = AgentActions.TURN
            elif (self.x, self.y) != (next_x, next_y): # Different position, perform move
                self.action = AgentActions.MOVE
            else:
                self.action = AgentActions.WAIT # Next position same as current position, wait
    
    def update_timestamp(self, current_time: int):
        """Update timestep"""
        self.current_timestamp = current_time
    
    def is_idle(self) -> bool:
        """Check if AGV is idle"""
        return self.status == AgentStatus.IDLE
    
    def is_moving_to_idle(self) -> bool:
        """Check if AGV is moving to idle point"""
        return self.status == AgentStatus.MOVING_TO_IDLE
    
    def is_loaded(self) -> bool:
        """Check if AGV is carrying cargo"""
        return self.loaded
    
    def at_edge(self) -> bool:
        """Check if AGV is at map edge"""
        return self.x == 1 or self.x == self.map_size[0] or self.y == 1 or self.y == self.map_size[1]

    def get_current_task(self) -> Optional[Task]:
        """Get AGV current task"""
        return self.current_task
    
    def assign_task(self, task: Task):
        """Assign task to AGV"""
        task.assign_to(self.id)
        self.current_task = task
        self.current_destination = task.get_pickup_points() + (None,)
        self.status = AgentStatus.MOVING_TO_PICKUP

    def clear_assign_task(self):
        """Clear current task assignment information for available AGV for next round allocation"""
        if self.status == AgentStatus.MOVING_TO_PICKUP:
            self.current_task = None
            self.current_destination = None
            self.status = AgentStatus.IDLE
    
    def load_task(self):
        """AGV pickup cargo"""
        self.loaded = True
        self.status = AgentStatus.MOVING_TO_DELIVERY
        self.current_task.picked_by(self.id, self.current_timestamp)
        
        delivery_dis = []
        for (x, y) in self.current_task.get_delivery_points():
            dis = self.map_path_cost[str(((self.x, self.y, self.pitch), (x, y)))]
            delivery_dis.append((dis, (x, y)))
        nearest_point = min(delivery_dis, key=lambda x: (x[0], x[1]))[1]

        self.current_destination = nearest_point + (None,)

    def unload_task(self) -> Task:
        """AGV unload cargo"""
        self.loaded = False
        task = self.current_task
        task.unload(self.current_timestamp)
        self.current_task = None
        self.current_destination = None
        self.status = AgentStatus.IDLE
        return task
    
    def set_status(self, status: AgentStatus):
        """Set AGV status"""
        self.status = status

    def set_destination(self, destination: Tuple[int, int, Optional[int]]):
        """Set AGV target point and orientation"""
        self.current_destination = destination

    def get_destination(self) -> Optional[Tuple[int, int, Optional[int]]]:
        """Get AGV current target point and orientation"""
        return self.current_destination

    def set_path(self, path: List[Tuple[int, int, int]]):
        """Set AGV path"""
        self.path = path

    def get_path(self) -> List[Tuple[int, int, int]]:
        """Get AGV path"""
        return self.path

    def get_remaining_path_length(self) -> int:
        """Get remaining timestep length until current AGV status ends"""
        return len(self.path) + 1 if self.status in [AgentStatus.MOVING_TO_PICKUP, AgentStatus.MOVING_TO_DELIVERY] else len(self.path)

    def record_trajectory(self):
        """Record AGV trajectory"""
        self.trajectory.append({
            'timestamp': self.current_timestamp,
            'name': self.id,
            'X': self.x,
            'Y': self.y,
            'pitch': self.pitch,
            'loaded': self.loaded,
            'task_id': self.current_task.id if self.loaded and self.current_task else '',
            'destination': self.current_task.get_destination_name() if self.loaded else '',
            'Emergency': self.current_task.is_urgent() if self.loaded else False,
        })