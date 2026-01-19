import csv
import sys
import traceback
import time
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import heapq
from DataLoader import DataLoader
from Agent import Agent, AgentStatus
from Task import Task
from PickupStation import PickupStation
from PathUtils import PathUtils
from evaluate import evaluate

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
        
        # Calculate endpointsulate endpoints
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

class TokenPassing:
    """TP algorithm implementation"""
    
    def __init__(self, config: MAPDConfig, tasks_data: List[Dict]):
        self.config = config
        self.tasks_data = tasks_data
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.induction_stations: Dict[str, PickupStation] = {}
        self.current_timestep = 0
        self.completed_tasks: List[Task] = []

        cache_path = "cache/" + DataLoader.get_map_type() + "_path_cost_dict.json"
        need_generate = False
        if not need_generate and self.load_from_json(cache_path):
            print("Using saved data")
        else:
            print("Please generate map path data!")
        self.path_false_count = 0

        # Initialize agents and taskse agents and tasks
        self._initialize_agents()
        self._initialize_stations()
        self._initialize_tasks()

        self.reservation_table = {}  # {timestep: {(x, y): agv_id}}  # Reservation table for conflict detection
        self.reservation_table[0] = {}
        for agv_id in list(self.agents.keys()):
            pos = self.agents[agv_id].get_position()
            self.reservation_table[0][(pos[0], pos[1])] = agv_id
        
        self.agent_initial_positions = [agent.get_position() for agent in self.agents.values()]  # Initial position list

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
                print(f"Warning: Cannot find induction station {task_data['start_point']}")
    
    def load_from_json(self, filename: str) -> bool:
        """Load dictionary from JSON file"""
        try:
            if Path(filename).exists():
                with open(filename, 'r', encoding='utf-8') as f:
                    self.map_path_cost = json.load(f)
                print(f"Data loaded from {filename}")
                return True
            else:
                print(f"File {filename} does not exist")
                return False
        except Exception as e:
            print(f"Load failed: {e}")
            return False
    
    def get_available_tasks(self) -> List[Task]:
        """Get assignable tasks"""
        current_tasks: List[Task] = []
        for induction in self.induction_stations.values():
            if induction.first_task_assigned:
                continue
            if induction.has_tasks():
                current_tasks.append(induction.pending_tasks[0])

        available_tasks = []
        
        for task in current_tasks:
            # Check if other agents' path endpoints are at task's pickup or delivery points
            pickup_occupied = False
            delivery_occupied = False
            
            for agent in self.agents.values():
                if agent.path:
                    end_location = (agent.path[-1][0], agent.path[-1][1])
                    if end_location == task.get_pickup_points():
                        pickup_occupied = True
                    if end_location in task.get_delivery_points():
                        delivery_occupied = True
            
            if not pickup_occupied and not delivery_occupied:
                available_tasks.append(task)
        
        return available_tasks
    
    def find_path_to_endpoint(self, agent: Agent, 
                              start,
                              goal,
                              start_timestamp: int,
                              start_pitch: int,
                              end_pitch: int,
                              status: Optional[AgentStatus]=None):
        """Find path from agent's current position to endpoint"""
        obstacles = self.config.obstacles.copy()
        for ag in self.agents.values():
            if ag.id == agent.id:
                continue
            if ag.is_idle():
                obstacles.append(ag.get_position())
            else:
                obstacles.append(ag.path[-1][:2])
        self.clear_reservation_table([agent.id], start_timestamp)
        path = self.space_time_astar(
            agent_id=agent.id,
            start=start,
            goal=goal,
            start_timestamp=start_timestamp,
            start_pitch=start_pitch,
            obstacles=obstacles,
            end_pitch=end_pitch,
            status=status
        )

        return path

    def path_plan(self, agent: Agent):
        """Plan path to pickup and delivery points"""
        # Plan path to pickup point
        final_path = []
        pickup_point = agent.get_current_task().get_pickup_points()
        path = self.find_path_to_endpoint(agent=agent, 
                                   start=agent.get_position(),
                                   goal=pickup_point,
                                   start_timestamp=self.current_timestep,
                                   start_pitch=agent.pitch,
                                   end_pitch=None, 
                                   status=AgentStatus.MOVING_TO_PICKUP)
        
        if path is None:
            self.path_false_count += 1
            return [], False
        else:
            final_path.extend(path)

        # Plan path to delivery point
        pickup_state = final_path[-1]
        delivery_dis = []
        for (x, y) in agent.get_current_task().get_delivery_points():
            dis = self.map_path_cost[str((pickup_state, (x, y)))]
            delivery_dis.append((dis, (x, y)))
        nearest_point = min(delivery_dis, key=lambda x: (x[0], x[1]))[1]

        path = self.find_path_to_endpoint(agent=agent, 
                                   start=pickup_state[:2],
                                   goal=nearest_point,
                                   start_timestamp=self.current_timestep + len(final_path),
                                   start_pitch=pickup_state[2],
                                   end_pitch=None, 
                                   status=AgentStatus.MOVING_TO_DELIVERY)
        
        if path is None:
            self.path_false_count += 1
            return [], False
        else:
            final_path.extend(path)

        available_endpoints = self.get_available_endpoints(agent, final_path)

        # Select nearest endpoint
        delivery_state = final_path[-1]
        best_endpoint = None
        min_distance = float('inf')
        
        for endpoint in available_endpoints:
            cost = self.map_path_cost[str((delivery_state, endpoint))]
            if cost < min_distance:
                min_distance = cost
                best_endpoint = endpoint

        path = self.find_path_to_endpoint(agent=agent, 
                                   start=delivery_state[:2],
                                   goal=best_endpoint,
                                   start_timestamp=self.current_timestep + len(final_path),
                                   start_pitch=delivery_state[2],
                                   end_pitch=None, 
                                   status=AgentStatus.MOVING_TO_IDLE)
        
        if path is None:
            self.path_false_count += 1
            return [], False
        else:
            final_path.extend(path)

        return final_path, True
        
    def get_available_endpoints(self, agent: Agent, current_path):
        """Get available endpoints for agent parking"""
        available_endpoints = []
        endpoints = list(set(self.config.non_task_endpoints.copy() + self.agent_initial_positions.copy()))
        for endpoint in endpoints:
            endpoint_free = True
            if endpoint in self.config.obstacles:
                continue
            for other_agent in self.agents.values():
                if other_agent.id == agent.id:
                    continue
                
                if other_agent.is_idle() and other_agent.get_position() == endpoint:
                    endpoint_free = False
                    break

                if other_agent.path and other_agent.path[-1][:2] == endpoint:
                    endpoint_free = False
                    break

                if other_agent.path:
                    p = [pos[:2] for pos in other_agent.path]
                    if endpoint in p:
                        cur_ag_distance = len(current_path) + self.map_path_cost[str((current_path[-1], endpoint))]
                        oth_ag_distance = p.index(endpoint)
                        if oth_ag_distance >= cur_ag_distance:
                            endpoint_free = False
                            break

            if endpoint_free:
                available_endpoints.append(endpoint)
        
        if not available_endpoints:
            raise ValueError(f"No available endpoints for AGV {agent.id} to park")
        return available_endpoints
    
    def goto_nearest_endpoint(self, agent: Agent):
        """Plan path to nearest endpoint for parking"""
        available_endpoints = self.get_available_endpoints(agent)

        # Select nearest endpoint
        current_state = (agent.x, agent.y, agent.pitch)
        best_endpoint = None
        min_distance = float('inf')
        
        for endpoint in available_endpoints:
            cost = self.map_path_cost[str((current_state, endpoint))]
            if cost < min_distance:
                min_distance = cost
                best_endpoint = endpoint

        path = self.find_path_to_endpoint(agent=agent, 
                                   start=agent.get_position(),
                                   goal=best_endpoint,
                                   start_timestamp=self.current_timestep,
                                   start_pitch=agent.pitch,
                                   end_pitch=None, 
                                   status=AgentStatus.MOVING_TO_IDLE)
        
        if path is None:
            raise ValueError(f"Unable to plan path for {agent.id} at {self.current_timestep} from {agent.get_position()} to {best_endpoint}, status: {AgentStatus.MOVING_TO_IDLE}, current task: {agent.get_current_task().id if agent.get_current_task() else 'no task'}")
        
        agent.current_task.assigned_agent_id = None
        induction = self.induction_stations[agent.current_task.get_induction_station_id()]
        induction.first_task_assigned = False
        agent.current_task = None
        agent.path = path
        agent.set_status(AgentStatus.MOVING_TO_IDLE)
        agent.set_destination(best_endpoint + (None,))
    
    def space_time_astar(self, 
                         agent_id:str, 
                         start:Tuple[int, int], 
                         goal:Tuple[int, int], 
                         start_timestamp:int, 
                         start_pitch:int, 
                         obstacles:List[Tuple[int, int]],
                         end_pitch:Optional[int]=None, 
                         status:Optional[AgentStatus]=None):
        """Space-time A* path planning algorithm"""
        directions = {
            'right': (1, 0),    # Right
            'up': (0, 1),    # Up
            'left': (-1, 0),  # Left
            'down': (0, -1),   # Down
            'stay': (0, 0)  # Stay
        }
        move_angles = {
            'right': 0,
            'up': 90,
            'left': 180,
            'down': 270
        }
        start_state = (start[0], start[1], start_pitch, 0)
        frontier = []
        g0 = 0  # 到达当前状态已花费的时间
        h0 = self.map_path_cost[str(((start[0], start[1], start_pitch), (goal[0], goal[1])))]
        heapq.heappush(frontier, (g0 + h0, h0, g0, start_state, []))
        visited = {}
        visited[start_state] = g0 + h0
        while frontier:
            f, h, g, state, path = heapq.heappop(frontier)
            if len(path) > 2 * (self.config.map_size[0] + self.config.map_size[1] + 1):  # Exceeds maximum possible path length, abandon
                print(f"Path length exceeds maximum, exiting path planning!")
                break
            x, y, pitch, t = state
            if (x, y) == (goal[0], goal[1]):
                if status in [AgentStatus.MOVING_TO_PICKUP, AgentStatus.MOVING_TO_DELIVERY]:
                    if self.have_vertex_conflict(x, y, g + PathUtils.LOAD_UNLOAD_TIME + start_timestamp, agent_id):
                        continue
                    if self.have_vertex_conflict(x, y, g + PathUtils.LOAD_UNLOAD_TIME + start_timestamp + 1, agent_id):
                        continue
                    path.append((x, y, pitch))
                if end_pitch is not None and pitch != end_pitch:
                    if self.have_vertex_conflict(x, y, g + PathUtils.TURN_COST + start_timestamp, agent_id):
                        continue
                    path.append((x, y, end_pitch))
                return path
            for action in ['right', 'up', 'left', 'down', 'stay']:
                dx, dy = directions[action]
                new_x, new_y = x + dx, y + dy
                new_pitch = pitch if action == 'stay' else move_angles[action]
                if (new_x < 1 or new_x > self.config.map_size[0] or 
                    new_y < 1 or new_y > self.config.map_size[1] or 
                    (new_x, new_y) in obstacles):
                    continue
                if action in ['right', 'up', 'left', 'down']:
                    if new_pitch == pitch:
                        new_g = g + 1
                        if self.have_vertex_conflict(new_x, new_y, new_g + start_timestamp, agent_id):
                            continue
                        if self.have_swap_conflict(x, y, g + start_timestamp, new_x, new_y):
                            continue
                        new_path = path + [(new_x, new_y, new_pitch)]
                    else:
                        new_g = g + PathUtils.TURN_COST + 1
                        if self.have_vertex_conflict(x, y, g + PathUtils.TURN_COST + start_timestamp, agent_id):
                            continue
                        if self.have_vertex_conflict(new_x, new_y, new_g + start_timestamp, agent_id):
                            continue
                        if self.have_swap_conflict(x, y, g + PathUtils.TURN_COST + start_timestamp, new_x, new_y):
                            continue
                        new_path = path + [(x, y, new_pitch), (new_x, new_y, new_pitch)]
                else:  # stay
                    new_g = g + 1
                    if self.have_vertex_conflict(new_x, new_y, new_g + start_timestamp, agent_id):
                        continue
                    new_path = path + [(new_x, new_y, new_pitch)]
                new_h = self.map_path_cost[str(((new_x, new_y, new_pitch), (goal[0], goal[1])))]
                new_state = (new_x, new_y, new_pitch, new_g)
                if new_state not in visited or visited[new_state] > new_g + new_h:
                    visited[new_state] = new_g + new_h
                    heapq.heappush(frontier, (new_g + new_h, new_h, new_g, new_state, new_path))
        return None  # No feasible path

    def have_vertex_conflict(self, 
                              x: int, 
                              y: int, 
                              t: int, 
                              agent_id: Optional[str] = None):
        """Check for vertex conflicts"""
        if t in self.reservation_table and (x, y) in self.reservation_table[t]:
            if agent_id is None or self.reservation_table[t][(x, y)] != agent_id:
                return True
        idle_agents_position = [agent.get_position() for agent in self.agents.values() if agent.id != agent_id and agent.is_idle()]
        if (x, y) in idle_agents_position:
            return True
        return False

    def have_swap_conflict(self, 
                            x: int, 
                            y: int, 
                            t: int, 
                            new_x: int, 
                            new_y: int):
        """Check for swap conflicts"""
        new_t = t + 1
        if t in self.reservation_table and (new_x, new_y) in self.reservation_table[t]:
            other_agent_id_1 = self.reservation_table[t][(new_x, new_y)]
            if new_t in self.reservation_table and (x, y) in self.reservation_table[new_t]:
                other_agent_id_2 = self.reservation_table[new_t][(x, y)]
                if other_agent_id_1 == other_agent_id_2:
                    return True
        return False

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
    
    def process_agent_request(self, agent: Agent):
        """Process agent's token request"""
        if agent.get_current_task():
            return
        
        # Get available tasks
        available_tasks = self.get_available_tasks()
        
        if available_tasks:
            # Select task with minimum heuristic cost
            best_task = None
            min_cost = float('inf')
            
            for task in available_tasks:
                # Calculate heuristic cost (distance from current position to pickup point)
                cost = self.map_path_cost[str(((agent.x, agent.y, agent.pitch), task.get_pickup_points()))]
                if cost < min_cost:
                    min_cost = cost
                    best_task = task

            # Plan path
            agent.current_task = best_task
            path, have_path = self.path_plan(agent)

            if have_path:
                # Assign task
                agent.assign_task(best_task)
                agent.path = path
                self.reserve_path(agent.id, path, self.current_timestep)
                induction = self.induction_stations.get(best_task.departure)
                induction.assign_first_task()
            else:
                agent.current_task = None
                if agent.status != AgentStatus.MOVING_TO_IDLE:
                    agent.path.append((agent.x, agent.y, agent.pitch))
        else:
            if agent.status != AgentStatus.MOVING_TO_IDLE:
                agent.path.append((agent.x, agent.y, agent.pitch))
    
    def run_timestep(self):
        """Run one timestep"""
        # Process each idle agent's request
        for agent in self.agents.values():
            if not agent.get_current_task():
                self.process_agent_request(agent)

        self.current_timestep += 1
        for station in self.induction_stations.values():
            station.update_timestamp(self.current_timestep)
        for agent in self.agents.values():
            agent.update_timestamp(self.current_timestep)
        print(f"Timestep advanced to {self.current_timestep}")

        # All agents move one step
        for agent in self.agents.values():
            if agent.path:
                cur_pos = agent.get_position()
                cur_state = (agent.x, agent.y, agent.pitch)
                next_state = agent.path.pop(0)
                
                if agent.current_task:
                    # Arrive at pickup point
                    if agent.status == AgentStatus.MOVING_TO_PICKUP and \
                        cur_pos == agent.current_task.get_pickup_points() and \
                        next_state == cur_state:
                        agent.load_task()
                        induction = self.induction_stations.get(agent.get_current_task().get_induction_station_id())
                        cur_task = induction.load_next_task()
                        if cur_task is None:
                            print(f"Error: Induction station {induction.id} has no tasks available, current Agent: {agent.id} task: {agent.get_current_task().id}")
                        if cur_task.id != agent.get_current_task().id:
                            print(f"Error: AGV {agent.id} picked wrong task {cur_task.id} != {agent.get_current_task().id}")
                    
                    # Arrive at delivery point
                    elif agent.status == AgentStatus.MOVING_TO_DELIVERY and \
                        cur_pos in agent.current_task.get_delivery_points() and \
                        next_state == cur_state:
                        task = agent.unload_task()
                        induction = self.induction_stations.get(task.get_induction_station_id())
                        induction.task_completed(task)
                        agent.status = AgentStatus.MOVING_TO_IDLE
                        self.completed_tasks.append(task)

                elif len(agent.path) == 0 and agent.status == AgentStatus.MOVING_TO_IDLE:
                    agent.set_status(AgentStatus.IDLE)

                agent.x, agent.y, agent.pitch = next_state
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
            print(f"Starting TP algorithm, {len(self.tasks_data)} tasks, {len(self.agents)} agents")
            
            start_time = time.time()

            while not self.all_tasks_completed() and self.current_timestep < max_timesteps:
                self.run_timestep()

            end_time = time.time()
            
            if self.all_tasks_completed():
                print(f"TP algorithm completed all tasks, time taken {end_time - start_time:.2f} seconds")
            else:
                print(f"TP algorithm reached max timesteps {max_timesteps}, not all tasks completed")
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
        print(f"Total completed tasks: {len(self.completed_tasks)}")
        print(f"Average service time: {service_time / len(self.completed_tasks):.2f}s")
        print(f"Path planning failures: {self.path_false_count}")
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
    
    # Run TP algorithm
    print("\n=== Running TP algorithm ===")
    tp = TokenPassing(config, tasks_data)
    tp.run()

if __name__ == "__main__":
    run_baseline()