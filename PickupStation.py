from typing import List, Optional, Deque, Tuple
from collections import deque
from Task import Task

class PickupStation:
    def __init__(self, station_id: str, pickup_point: Tuple[int, int]):
        self.id: str = station_id
        self.pickup_point: Tuple[int, int] = pickup_point  # Pickup point coordinates (x, y)
        self.pending_tasks: List[Task] = []  # Pending task queue
        self.finished_tasks: List[Task] = []  # Completed task list
        self.last_pickup_timestamp: Optional[int] = None  # Timestamp of last task pickup
        self.current_timestamp: int = 0  # Current timestamp

        self.allocate_list: List[Task] = []  # Allocation list
        self.urgent_task_count_list: List[int] = []  # Urgent task count statistics
        self.task_urgency_list: List[float] = []  # Urgent task average remaining time statistics
        self.allocate_index: int = 0  # Allocation index
        self.first_task_assigned: bool = False  # Whether first task has been assigned

    def get_pickup_point(self) -> Tuple[int, int]:
        """Get induction station pickup point coordinates"""
        return self.pickup_point
        
    def update_timestamp(self, current_time: int):
        """Update timestep"""
        self.current_timestamp = current_time
        
    def add_task(self, task: Task):
        """Add task to queue"""
        if task.departure != self.id:
            raise ValueError(f"Task {task.id} does not belong to induction station {self.id}")
        self.pending_tasks.append(task)
        
    def add_tasks_batch(self, tasks: List[Task]):
        """Batch add tasks"""
        for task in tasks:
            self.add_task(task)
    
    def task_completed(self, task: Task):
        """Mark task as completed"""
        if task.departure != self.id:
            raise ValueError(f"Task {task.id} does not belong to induction station {self.id}")
        self.finished_tasks.append(task)

    def has_tasks(self) -> bool:
        """Check if there are remaining tasks"""
        return len(self.pending_tasks) > 0

    def is_complete(self) -> bool:
        """Check if all tasks are completed"""
        return len(self.pending_tasks) == 0
    
    def load_next_task(self) -> Optional[Task]:
        """Pick up and load next task"""
        if not self.has_tasks():
            return None
        task = self.pending_tasks.pop(0)
        if len(self.pending_tasks) > 0:
            self.pending_tasks[0].available_timestamp = self.current_timestamp + 1
        self.last_pickup_timestamp = self.current_timestamp
        self.first_task_assigned = False
        return task
    
    def assign_first_task(self):
        """Check if first task has been assigned"""
        self.first_task_assigned = True

    # Task allocation related functions
    def start_allocation(self):
        """Start task allocation, reset index"""
        self.allocate_list = self.pending_tasks.copy()
        count = 0
        for i, task in enumerate(self.allocate_list):
            count += 1
            if task.is_urgent():
                avg_remain_time = float(task.remaining_time - self.current_timestamp) / (i + 1)
                for _ in range(count):
                    self.task_urgency_list.append(avg_remain_time)
                    self.urgent_task_count_list.append(i + 1)
                count = 0
            elif i == len(self.allocate_list) - 1:
                for _ in range(count):
                    self.task_urgency_list.append(-1)
                    self.urgent_task_count_list.append(-1)

    def has_tasks_to_allocate(self) -> bool:
        """Check if there are tasks pending allocation"""
        return len(self.allocate_list) > 0 and self.allocate_index < len(self.allocate_list)

    def get_next_task(self) -> Task:
        """Get next task during task allocation process"""
        task = self.allocate_list[self.allocate_index]
        self.allocate_index += 1
        return task

    def end_allocation(self):
        """End task allocation, clear allocation list"""
        self.allocate_list = []
        self.task_urgency_list = []
        self.urgent_task_count_list = []
        self.allocate_index = 0

    def get_task_to_be_allocated_count(self) -> int:
        """Query total number of tasks to be allocated"""
        return len(self.allocate_list)
    
    def get_remaining_task_count(self) -> int:
        """Query number of remaining tasks to be allocated"""
        return len(self.allocate_list)-self.allocate_index
    
    def get_allocated_task_count(self) -> int:
        """Query number of allocated tasks"""
        return self.allocate_index
    
    def is_first_task(self) -> bool:
        """Query whether current allocated task is the first task"""
        return self.allocate_index == 0
        
    def has_urgent_tasks(self) -> bool:
        """Query if there are urgent tasks after allocation index"""
        return any(task.is_urgent() for task in self.allocate_list[self.allocate_index:])
        
    def get_task_urgency_from_index(self) -> float:
        """Query remaining number of urgent tasks at index position"""
        return self.task_urgency_list[self.allocate_index]
    
    def get_urgent_task_count_from_index(self) -> int:
        """Query remaining number of urgent tasks at index position"""
        return self.urgent_task_count_list[self.allocate_index]

    def get_time_since_last_pickup(self) -> int:
        """Query number of timesteps since last task was picked up"""
        if self.last_pickup_timestamp is None:
            return -1  # Indicates no task has ever been picked up
        return self.current_timestamp - self.last_pickup_timestamp
    
