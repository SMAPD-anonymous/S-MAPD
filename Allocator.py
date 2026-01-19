from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from Task import Task
from Agent import Agent
from PickupStation import PickupStation
from scipy.optimize import linear_sum_assignment
import numpy as np

class BSA:
    """Batch Sequential Assignment"""
    def __init__(self, agents: List[Agent], induction_stations: List[PickupStation], obstacles: List[Tuple[int, int]], map_size: Tuple[int, int], map_cost):
        self.agents: List[Agent] = agents
        self.agents.sort(key=lambda a: int(a.id[3:]))  # Sort by ID to ensure consistent allocation order
        self.induction_stations = induction_stations
        self.obstacles = obstacles
        self.map_size = map_size  # Map size (width, height)
        self.map_cost = map_cost  # Map path cost
        self.weights = {
            'task_to_be_allocated_count': 1.0,   # Task count factor weight
            'allocate_task_count': 1.0, # Allocation count factor weight
        }

    def allocate_tasks(self) -> Optional[List[Dict[str, any]]]:
        """Allocate tasks to idle AGVs using multi-factor sorting algorithm"""
        available_agents = [agent for agent in self.agents if not agent.is_loaded()]
        if not available_agents:
            return None
        
        for station in self.induction_stations:
            station.start_allocation()  # Enable induction station task allocation mode

        # # Replace BSA with Greedy Selection
        # available_stations = [station for station in self.induction_stations if station.has_tasks_to_allocate()]
        # allocate_results = []
        # allocate_tasks = {}
        # allocate_agents = {}
        # for agent in available_agents:
        #     best_station = None
        #     min_distance = float('inf')
        #     for station in available_stations:
        #         dis = self.map_cost[str(((agent.x, agent.y, agent.pitch), station.pickup_point))]
        #         if dis < min_distance:
        #             min_distance = dis
        #             best_station = station
        #     if best_station:
        #         task = best_station.get_next_task()
        #         allocate_tasks.setdefault(best_station.id, []).append(task)
        #         allocate_agents.setdefault(best_station.id, []).append((agent.id, min_distance))
        #         if not best_station.has_tasks_to_allocate():
        #             available_stations = [s for s in available_stations if s.id != best_station.id]
        #     else:
        #         break
        # for station_id in allocate_tasks.keys():
        #     tasks = allocate_tasks[station_id]
        #     l = allocate_agents[station_id]
        #     l.sort(key=lambda x: (x[1], int(x[0][3:])))  # 按距离排序
        #     for i, task in enumerate(tasks):
        #         agent_id = l[i][0]
        #         allocate_results.append({"agent_id": agent_id, "task": task})
        
        # Continuously score all induction stations and select the best one, get its next task
        agents = self.agents
        tasks_to_allocate = []
        while len(tasks_to_allocate) < len(agents):
            next_task = self.get_best_task()
            if not next_task:
                break
            tasks_to_allocate.append(next_task)

        for station in self.induction_stations:
            station.end_allocation()  # End induction station task allocation mode
        
        allocate_results = self.match_tasks(tasks_to_allocate)
        return allocate_results

    def get_best_task(self) -> Optional[Task]:
        """Find the best task for allocation"""
        scores = []
        station_stats = self.collect_station_statistics()  # Collect all induction station statistics for normalization in current allocation round
        for station in self.induction_stations:
            if not station.has_tasks_to_allocate():
                continue
            score_details = self.calculate_station_score_details(station, station_stats)  # Calculate score of current task at this induction station
            total_score = score_details['task_to_be_allocated_count'] * self.weights['task_to_be_allocated_count'] - \
                            score_details['allocate_task_count'] * self.weights['allocate_task_count']
            scores.append({
                'station': station,
                'score': total_score,
                'details': score_details
            })
        if scores:
            scores.sort(key=lambda x: (x['score'], int(x['station'].id[2:])), reverse=True)
            best_station: PickupStation = scores[0]['station']
            best_task: Task = best_station.get_next_task()
            return best_task
        return None

    def collect_station_statistics(self) -> Dict[str, any]:
        """Collect all induction station statistics for normalization"""
        stats = {
            'task_to_be_allocated_counts': [],
            'allocate_task_counts': [],
        }
        for station in self.induction_stations:
            if station.has_tasks_to_allocate():
                stats['task_to_be_allocated_counts'].append(station.get_task_to_be_allocated_count())
                stats['allocate_task_counts'].append(station.get_allocated_task_count())
        return stats

    def calculate_station_score_details(self, station: PickupStation,
                              station_stats: Dict[str, any]) -> Dict[str, float]:
        """Calculate detailed score for each factor"""
        details = {}
        details['task_to_be_allocated_count'] = self.calculate_task_to_be_allocated_count_score(station, station_stats)  # Task count factor
        details['allocate_task_count'] = self.calculate_allocate_count_score(station, station_stats)  # Allocation count factor
        return details

    def calculate_task_to_be_allocated_count_score(self, station: PickupStation, station_stats: Dict[str, any]) -> float:
        """Calculate task count factor score"""
        if not station_stats['task_to_be_allocated_counts']:
            return 0.0
        current_count = station.get_task_to_be_allocated_count()
        sum_count = sum(station_stats['task_to_be_allocated_counts'])
        if sum_count == 0:
            return 0.0
        return current_count / sum_count
    
    def calculate_allocate_count_score(self, station: PickupStation, station_stats: Dict[str, any]) -> float:
        """Calculate allocation count factor score"""
        if not station_stats['allocate_task_counts']:
            return 0.0
        current_count = station.get_allocated_task_count()
        sum_count = sum(station_stats['allocate_task_counts'])
        if sum_count == 0:
            return 0.0
        return current_count / sum_count

    def match_tasks(self, tasks_to_allocate: List[Task]):
        """Allocate tasks to AGVs"""
        allocate_results = []
        agents = self.agents
        cost_matrix = np.zeros((len(tasks_to_allocate), len(agents)))
        for i, task in enumerate(tasks_to_allocate):
            for j, agent in enumerate(agents):
                if agent.is_loaded():
                    remain_distance = agent.get_remaining_path_length()  # Remaining path distance for loaded AGVs
                    if remain_distance <= 1:
                        end_position = agent.get_position()
                        end_pitch = agent.get_pitch()
                    else:
                        end_position = agent.get_path()[-1][:2]
                        end_pitch = agent.get_path()[-1][2]
                    distance = self.map_cost[str(((end_position[0], end_position[1], end_pitch), task.get_pickup_points()))]
                    cost_matrix[i][j] = distance + remain_distance * 1.0  # Consider remaining path distance for loaded AGVs, increase weight appropriately
                else:
                    distance = self.map_cost[str(((agent.x, agent.y, agent.pitch), task.get_pickup_points()))]
                    cost_matrix[i][j] = distance
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        row_ind, col_ind = list(row_ind), list(col_ind)
        # Group tasks by induction station
        station_tasks = defaultdict(list)
        for induction in self.induction_stations:
            for task_idx, task in enumerate(tasks_to_allocate):
                station_id = task.get_induction_station_id()
                if station_id == induction.id:
                    if induction.id in station_tasks:
                        station_tasks[induction.id].append((task_idx, col_ind[task_idx], agents[col_ind[task_idx]]))
                    else:
                        station_tasks[induction.id] = [(task_idx, col_ind[task_idx], agents[col_ind[task_idx]])]
        for station_id, assignment in station_tasks.items():
            station_agent_distance = []
            for task_idx, agent_idx, agent in assignment:
                distance = cost_matrix[task_idx][agent_idx]
                station_agent_distance.append((distance, agent.id, agent_idx))
            station_agent_distance.sort(key=lambda x: (x[0], int(x[1][3:])))  # Sort by distance
            station_task_idx = [x[0] for x in assignment]
            for i, task_idx in enumerate(station_task_idx):
                col_ind[task_idx] = station_agent_distance[i][2]
            
        for task_idx, agent_idx in zip(row_ind, col_ind):
            task = tasks_to_allocate[task_idx]
            agent = agents[agent_idx]
            if not agent.is_loaded():
                allocate_results.append({"agent_id": agent.id, "task": task})

        return allocate_results