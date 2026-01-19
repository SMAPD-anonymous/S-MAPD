import csv
import sys
import traceback
import time
from typing import List, Dict, Tuple
from pathlib import Path
import json
from DataLoader import DataLoader
from Agent import Agent, AgentStatus
from Task import Task
from PickupStation import PickupStation
from evaluate import evaluate
from CBS import CBSVariantsSolver

class MAPDConfig:
    """MAPD configuration"""
    def __init__(self, map_size: Tuple[int, int], obstacles: List[Tuple[int, int]], 
                 start_points: Dict[str, Tuple[int, int]], 
                 end_points: Dict[str, Tuple[int, int]],
                 agents: List[Dict]):
        self.map_size = map_size
        self.obstacles = obstacles
        self.start_points = start_points  # Induction station positions
        self.end_points = end_points      # Unload point positions
        self.agents_data = agents
        
        # Calculate endpoints
        self.task_endpoints = []
        for point in start_points.values():
            if point[0] == 1:
                pickup_point = (point[0] + 1, point[1])
            elif point[0] == map_size[0]:
                pickup_point = (point[0] - 1, point[1])
            self.task_endpoints.append(pickup_point)
        for point in end_points.values():
            delivery_points = [
                (point[0], point[1] + 1),  # Top
                (point[0], point[1] - 1),  # Bottom
                (point[0] - 1, point[1]),  # Left
                (point[0] + 1, point[1])   # Right
            ]
            for x, y in delivery_points:
                if 1 <= x <= map_size[0] and 1 <= y <= map_size[1] and (x, y) not in obstacles:
                    self.task_endpoints.append((x, y))
        self.non_task_endpoints = self._find_non_task_endpoints()
    
    def _find_non_task_endpoints(self) -> List[Tuple[int, int]]:
        """Find non-task endpoints (free positions on map edges)"""
        non_task_endpoints = []
        map_w, map_h = self.map_size
        
        # Check free positions on map edges
        for x in range(2, map_w):
            non_task_endpoints.append((x, 1))
            non_task_endpoints.append((x, map_h))
        
        for y in range(2, map_h):
            non_task_endpoints.append((1, y))
            non_task_endpoints.append((map_w, y))
        
        return non_task_endpoints

class CENTRAL:
    def __init__(self, config: MAPDConfig, tasks_data: List[Dict]):
        self.config = config
        self.tasks_data = tasks_data
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.induction_stations: Dict[str, PickupStation] = {}
        self.current_timestep = 0
        self.completed_tasks: List[Task] = []

        self.max_planning_time = 3600  # Max planning time in seconds

        cache_path = "cache/" + DataLoader.get_map_type() + "_path_cost_dict.json"
        need_generate = False
        if not need_generate and self.load_from_json(cache_path):
            print("Using saved data")
        else:
            print("Please generate map path data!")

        # Initialize agents and tasks
        self._initialize_agents()
        self._initialize_stations()
        self._initialize_tasks()

        self.reservation_table = {}  # {timestep: {(x, y): agv_id}}  # Reservation table for conflict detection
        self.reservation_table[0] = {}
        for agv_id in list(self.agents.keys()):
            pos = self.agents[agv_id].get_position()
            self.reservation_table[0][(pos[0], pos[1])] = agv_id

    def _initialize_agents(self):
        """Initialize agents"""
        for agent_data in self.config.agents_data:
            agent_id = agent_data["id"]
            x, y = agent_data["pos"]
            pitch = agent_data["pitch"]
            self.agents[agent_id] = Agent(agent_id, x, y, pitch, self.config.map_size, self.map_path_cost)
            self.agents[agent_id].record_trajectory()

    def _initialize_stations(self):
        """Initialize induction stations"""
        for name, coor in self.config.start_points.items():
            if coor[0] == 1:
                pickup_point = (coor[0] + 1, coor[1])
            else:
                pickup_point = (coor[0] - 1, coor[1])
            self.induction_stations[name] = PickupStation(
                station_id=name,
                pickup_point=pickup_point
            )
    
    def _initialize_tasks(self):
        """Initialize tasks"""
        for _, task_data in enumerate(self.tasks_data):
            task_id = task_data["task_id"]
            
            # Get pickup and delivery point coordinates
            start_point_name = task_data["start_point"]
            end_point_name = task_data["end_point"]
            end_point = self.config.end_points.get(end_point_name)

            task = Task(task_id=task_id, 
                        departure=start_point_name, 
                        pickup_point=self.induction_stations[task_data['start_point']].get_pickup_point(),
                        destination=end_point_name,
                        destination_coordinate=end_point, 
                        priority=task_data['priority'],
                        remaining_time=task_data.get('remaining_time'),
                        obstacles=self.config.obstacles,
                        map_size=self.config.map_size)
            self.tasks[task_id] = task

            station = self.induction_stations.get(task_data['start_point'])
            if station:
                station.add_task(task)
            else:
                print(f"警告: 找不到供件台 {task_data['start_point']}")
    
    def load_from_json(self, filename: str) -> bool:
        """从JSON文件加载字典"""
        try:
            if Path(filename).exists():
                with open(filename, 'r', encoding='utf-8') as f:
                    self.map_path_cost = json.load(f)
                print(f"数据已从 {filename} 加载")
                return True
            else:
                print(f"文件 {filename} 不存在")
                return False
        except Exception as e:
            print(f"加载失败: {e}")
            return False

    def plan_path_with_cbs(self, need_plan_agents):
        """使用CBS为所有智能体规划路径"""
        if not need_plan_agents:
            return

        obstacles = self.config.obstacles.copy()
        for agent in self.agents.values():
            if agent.is_idle():
                obstacles.append(agent.get_position())

        agent_info = {}
        for agent_id in need_plan_agents:
            agent = self.agents[agent_id]
            agent.path.clear()
            agent_info[agent_id] = {
                'start': agent.get_position(),
                'goal': agent.get_destination()[:2],
                'start_pitch': agent.pitch,
                'end_pitch': agent.get_destination()[2],
                'status': agent.status
            }

        planner = CBSVariantsSolver(conflict_agvs=agent_info,
                                    conflict_timestamp=self.current_timestep,
                                    reservation_table={},
                                    obstacles=obstacles,
                                    map_size=self.config.map_size,
                                    cost_map=self.map_path_cost,
                                    algorithm_type="GCBS_HL")
        
        paths, _ = planner.find_solution()
        for agent_id, path in paths.items():
            self.agents[agent_id].set_path(path[1:])
            self.reserve_path(agent_id, path[1:], self.current_timestep)
        
    def goto_nearest_endpoint(self, agent: Agent):
        """Get available endpoints for agent parking"""
        available_endpoints = []
        for endpoint in self.config.non_task_endpoints:
            endpoint_free = True
            if endpoint in self.config.obstacles:
                continue
            for other_agent in self.agents.values():
                if other_agent.id == agent.id:
                    continue
                    
                if other_agent.status == AgentStatus.MOVING_TO_IDLE and other_agent.get_destination()[:2] == endpoint:
                    endpoint_free = False
                    break

                if other_agent.is_idle() and other_agent.get_position() == endpoint:
                    endpoint_free = False
                    break

            if endpoint_free:
                available_endpoints.append(endpoint)
        
        if not available_endpoints:
            raise ValueError(f"No available endpoints for AGV {agent.id} to park")
        
        best_endpoint = None
        min_distance = float('inf')
        
        for endpoint in available_endpoints:
            cost = self.map_path_cost[str(((agent.x, agent.y, agent.pitch), endpoint))]
            if cost < min_distance:
                min_distance = cost
                best_endpoint = endpoint

        agent.set_status(AgentStatus.MOVING_TO_IDLE)
        agent.set_destination(best_endpoint + (None,))

    def reserve_path(self, 
                      agent_id: str, 
                      path: List[Tuple[int, int, int]], 
                      start_timestamp: int):
        """Reserve path for Agent in reservation table"""
        for t, (x, y, _) in enumerate(path, start=start_timestamp + 1):
            self.reservation_table.setdefault(t, {})[(x, y)] = agent_id

    def clear_reservation_table(self, 
                                agent_ids: List[str], 
                                timestep: int):
        """Clear all agents in list from reservation table after specified timestep"""
        keys_to_delete = [t for t in self.reservation_table if t > timestep]
        for t in keys_to_delete:
            for (x, y) in list(self.reservation_table[t].keys()):
                if self.reservation_table[t][(x, y)] in agent_ids:
                    del self.reservation_table[t][(x, y)]
    
    def get_task(self, agent: Agent):
        """Process agent's token request"""
        if agent.get_current_task():
            return
        # Get available tasks
        closest_induction = None
        min_distance = float('inf')
        for induction in self.induction_stations.values():
            if induction.has_tasks():
                distance = self.map_path_cost[str(((agent.x, agent.y, agent.pitch), induction.get_pickup_point()))]
                if distance < min_distance:
                    min_distance = distance
                    closest_induction = induction

        best_task = None
        assigned_agent = None
        if closest_induction:
            for t in closest_induction.pending_tasks:
                assigned_agent = None
                if t.is_assigned() and t.assigned_agent_id != agent.id:
                    assigned_agent = self.agents.get(t.assigned_agent_id)
                    assigned_agent_path = self.map_path_cost[str(((assigned_agent.x, assigned_agent.y, assigned_agent.pitch), t.get_pickup_points()))]
                    current_agent_path = self.map_path_cost[str(((agent.x, agent.y, agent.pitch), t.get_pickup_points()))]
                    if current_agent_path < assigned_agent_path:
                        best_task = t
                        break
                else:
                    best_task = t
                    break
        if best_task:
            # Assign task
            if assigned_agent is not None:
                assigned_agent.path.clear()
                assigned_agent.current_task = None
                assigned_agent.current_destination = None
                assigned_agent.set_status(AgentStatus.IDLE)

            agent.path.clear()
            agent.assign_task(best_task)

            if assigned_agent is not None:
                self.get_task(assigned_agent)

        else:
            agent.path.clear()
            if agent.get_position() not in self.config.non_task_endpoints:
                self.goto_nearest_endpoint(agent)

            else:
                agent.set_status(AgentStatus.IDLE)
                agent.path.append((agent.x, agent.y, agent.pitch))

    def run_timestep(self):
        """Run one timestep"""
        # Process each idle agent's request
        for agent in self.agents.values():
            if not agent.get_current_task():
                self.get_task(agent)

        need_replan_path = False
        for agent in self.agents.values():
            if len(agent.path) == 0:
                need_replan_path = True
                break
        if need_replan_path:
            self.clear_reservation_table([list(self.agents.keys())], self.current_timestep)
            moving_agents = [a.id for a in self.agents.values() if a.status != AgentStatus.IDLE]
            self.plan_path_with_cbs(moving_agents)

        self.current_timestep += 1
        for station in self.induction_stations.values():
            station.update_timestamp(self.current_timestep)
        print(f"Timestep advanced to {self.current_timestep}")

        # All agents move one step
        for agent in self.agents.values():
            if agent.path:
                cur_state = (agent.x, agent.y, agent.pitch)
                next_state = agent.path.pop(0)
                
                if len(agent.path) == 0:
                    # Arrive at pickup point
                    if agent.status == AgentStatus.MOVING_TO_PICKUP and \
                        cur_state[:2] == agent.current_task.get_pickup_points() and \
                        next_state == cur_state:
                        induction = self.induction_stations.get(agent.get_current_task().get_induction_station_id())
                        next_task = induction.load_next_task()
                        if next_task is None:
                            raise ValueError(f"Induction station {induction.id} has no tasks available, current Agent: {agent.id} task: {agent.get_current_task().id}")
                        if next_task.id != agent.get_current_task().id:
                            assigned_agent = self.agents.get(next_task.assigned_agent_id)
                            if assigned_agent:
                                assigned_agent.current_task = agent.current_task
                                assigned_agent.current_task.assign_to(assigned_agent.id)
                            next_task.assign_to(agent.id)
                            agent.current_task = next_task
                        agent.load_task()
                    
                    # Arrive at delivery point
                    elif agent.status == AgentStatus.MOVING_TO_DELIVERY and \
                        cur_state[:2] in agent.current_task.get_delivery_points() and \
                        next_state == cur_state:
                        task = agent.unload_task()
                        self.completed_tasks.append(task)

                    elif agent.status == AgentStatus.MOVING_TO_IDLE:
                        agent.set_status(AgentStatus.IDLE)
                        agent.path.append((agent.x, agent.y, agent.pitch))
                    
                    elif agent.status == AgentStatus.IDLE:
                        agent.path.append((agent.x, agent.y, agent.pitch))

                agent.x, agent.y, agent.pitch = next_state
                agent.update_timestamp(self.current_timestep)
            else:
                raise ValueError(f"AGV {agent.id} has no path to execute at timestep {self.current_timestep}!")
            
        for agent in self.agents.values():
            agent.record_trajectory()
    
    def all_tasks_completed(self) -> bool:
        """Check if all tasks are completed"""
        return len(self.completed_tasks) == len(self.tasks_data)
    
    def run(self, max_timesteps: int = 5000):
        """Run algorithm until all tasks completed or max timesteps reached"""
        try:
            print(f"Starting Central algorithm, {len(self.tasks_data)} tasks, {len(self.agents)} agents")
            
            start_time = time.time()

            while not self.all_tasks_completed() and self.current_timestep < max_timesteps:
                self.run_timestep()

                current_planning_time = time.time() - start_time
                if current_planning_time > self.max_planning_time:
                    raise KeyboardInterrupt(f"Planner runtime exceeded {self.max_planning_time} seconds, aborted")

            end_time = time.time()
            
            if self.all_tasks_completed():
                print(f"Central algorithm completed all tasks, time taken {end_time - start_time:.2f} seconds")
            else:
                print(f"Central algorithm reached max timesteps {max_timesteps}, not all tasks completed")
            self.generate_trajectory_csv()
            self.evaluate_trajectory_csv()
        except BaseException as e:
            self.generate_trajectory_csv()
            self.evaluate_trajectory_csv()
            exc_type, exc_value, exc_traceback = sys.exc_info()
            stack_summary = traceback.extract_tb(exc_traceback)
            last_frame = stack_summary[-1] if stack_summary else None
            print(f"⚠️ Exception occurred: {exc_type.__name__}: {exc_value}")
            if last_frame:
                print(f"📂 File: {last_frame.filename}")
                print(f"📍 Line: {last_frame.lineno}")
                print(f"📝 Code: {last_frame.line}")
                print(f"🧩 Function: {last_frame.name}")
    
    def generate_trajectory_csv(self):
        """Generate final CSV file"""
        status_snapshots: List[Dict[str, any]] = []
        for agent in self.agents.values():
            status_snapshots.extend(agent.trajectory)
        fieldnames = [
            'timestamp', 'name', 'X', 'Y', 'pitch', 
            'loaded', 'task_id', 'destination', 'Emergency'
        ]
        sorted_snapshots = sorted(
            status_snapshots,
            key=lambda x: (
                x['timestamp'],  # Primary sort key: timestamp
                x['name'].lower()  # Secondary sort key: Name first letter (case-insensitive)
            )
        )
        with open(DataLoader.get_trajectory_path(), 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for snapshot in sorted_snapshots:
                writer.writerow(snapshot)

    def evaluate_trajectory_csv(self):
        result = evaluate(
            agv_file=DataLoader.get_trajectory_path(),
            task_file=DataLoader.get_task_path(),
            map_file=DataLoader.get_map_path()
        )
        
        print("Evaluation results:")
        print(f"Trajectory validity: {'Passed' if result['trajectory_valid'] else 'Failed'}")
        if result['valid_completed_num'] == len(self.tasks_data):
            print("All tasks completed correctly")
        else:
            print(f"Correctly completed tasks: {result['valid_completed_num']} / {len(self.tasks_data)}")
            print(f"False pickup count: {result['false_pickup_count']}")
            for error in result['false_pickups']:
                print(f"Time {error['timestamp']}s, AGV {error['agv']} false pickup, details: {error['details']}")
        print(f"Total completion time: {result['total_completion_time']}s")
        service_time = 0
        for t in self.completed_tasks:
            service_time  += t.end_timestamp - t.available_timestamp
        print(f"Average service time: {service_time / len(self.completed_tasks):.2f}s")
        print(f"Collision count: {result['collision_count']}")
        
        if not result['trajectory_valid']:
            print("\nTrajectory error details:")
            for error in result['trajectory_errors']:
                print(f"[{error['type']}] {error['agv']} at {error['timestamp']}s: {error['details']}")
        
        if not result['collision_count'] == 0:
            print("\nCollision details:")
            for collision in result['collisions']:
                print(f"Time {collision['timestamp']}s, AGV {collision['AGVs']} at ({collision['X']}, {collision['Y']}) {collision['type']} collision")

def run_baseline():
    """Run baseline algorithm comparison"""

    # Get map elements
    start_points, end_points, agv_list, obstacle_list, map_size = DataLoader.get_map_elements()
    
    # Get task data
    tasks_data = DataLoader.get_tasks()

    obstacles = list(start_points.values()) + list(end_points.values()) + obstacle_list
    
    config = MAPDConfig(
        map_size=map_size,
        obstacles=obstacles,
        start_points=start_points,
        end_points=end_points,
        agents=agv_list
    )
    
    print(f"Map size: {map_size}")
    print(f"Start point count: {len(start_points)}")
    print(f"End point count: {len(end_points)}")
    print(f"AGV count: {len(agv_list)}")
    print(f"Task count: {len(tasks_data)}")
    
    # Run Central algorithm
    print("\n=== Running Central algorithm ===")
    central = CENTRAL(config, tasks_data)
    central.run()

if __name__ == "__main__":
    run_baseline()