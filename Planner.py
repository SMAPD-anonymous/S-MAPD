import time
import sys
import traceback
import json
from pathlib import Path
from csv import DictWriter
import heapq
from typing import List, Optional, Tuple, Dict, Any
from collections import deque
from Task import Task, TaskPriority
from Agent import Agent, AgentStatus, AgentActions
from PickupStation import PickupStation
from DataLoader import DataLoader
from Allocator import BSA
from PathUtils import PathUtils
from CBS import CBSVariantsSolver
from evaluate import evaluate

class Planner:
    """Planner, responsible for AGV scheduling and path planning"""
    def __init__(self):
        self.agents: Dict[str, Agent] = {}  # AGV dictionary
        self.induction_stations: Dict[str, PickupStation] = {}  # Induction station dictionary
        self.end_points: Dict[str, Tuple[int, int]] = {}  # Unload point coordinate dictionary
        self.obstacles: List[Tuple[int, int]] = []  # Obstacle list
        self.tasks: List[Task] = []  # Task list
        self.completed_tasks: List[Task] = []  # Completed task list
        self.current_timestamp = 0
        self.map_size: Tuple[int, int] = (0, 0)

        # Planner runtime monitoring
        self.max_planning_time = 36000

        self.initialize()

        self.reservation_table = {}  # {timestep: {(x, y): agv_id}}
        self.reservation_table[0] = {}
        for agv_id in list(self.agents.keys()):
            pos = self.agents[agv_id].get_position()
            self.reservation_table[0][(pos[0], pos[1])] = agv_id

        self.agents_initial_positions = [agent.get_position() for agent in self.agents.values()]

    def initialize(self):
        """Initialize planner, load data"""
        start_points, end_points, agv_list, obstacle_list, map_size = DataLoader.get_map_elements()
        self.map_size = map_size
        self.end_points = end_points
        task_data = DataLoader.get_tasks()
        obstacle_list = list(start_points.values()) + list(end_points.values()) + obstacle_list

        cache_path = "cache/" + DataLoader.get_map_type() + "_path_cost_dict.json"
        need_generate = False
        if not need_generate and self.load_from_json(cache_path):
            print("Using saved data")
        else:
            print("Generating new data...")
            start_time = time.time()
            self.generate_a_star_with_turn_cost_map()
            end_time = time.time()
            print(f"A* path cost map generation completed, time elapsed: {end_time - start_time:.2f} seconds")
            self.save_to_json(cache_path)
            print("New data generated and saved")

        self.initialize_agents(agv_list)
        self.initialize_induction_stations(start_points)
        self.initialize_obstacles(obstacle_list)
        self.load_tasks(task_data)

        print("Planner initialization completed")

    def initialize_agents(self, agv_list: List[Dict]):
        """Initialize AGVs"""
        for agv in agv_list:
            agent = Agent(
                agent_id=agv["id"],
                start_x=agv["pos"][0],
                start_y=agv["pos"][1],
                start_pitch=agv["pitch"],
                map_size=self.map_size,
                map_path_cost=self.map_path_cost
            )
            self.agents[agv["id"]] = agent
            agent.record_trajectory()  # Record initial position
        print("AGVs initialized")

    def initialize_induction_stations(self, station_data: Dict):
        """Initialize all induction stations"""
        for name, coor in station_data.items():
            if coor[0] == 1:
                pickup_point = (coor[0] + 1, coor[1])
            else:
                pickup_point = (coor[0] - 1, coor[1])
            self.induction_stations[name] = PickupStation(
                station_id=name,
                pickup_point=pickup_point
            )
        print("Induction stations initialized")

    def initialize_obstacles(self, obstacle_list: List[Tuple[int, int]]):
        """Initialize obstacle list"""
        self.obstacles = obstacle_list
        print("Obstacles initialized")

    def load_tasks(self, task_data: List[Dict]):
        """Load task data and assign to corresponding induction stations"""
        for data in task_data:
            task = Task(
                task_id=data['task_id'],
                departure=data['start_point'],
                pickup_point=self.induction_stations[data['start_point']].get_pickup_point(),
                destination=data['end_point'],
                destination_coordinate=self.end_points.get(data['end_point'], (0, 0)),
                priority=TaskPriority.HIGH if data.get('priority', None) == "Urgent" else TaskPriority.NORMAL,
                remaining_time=data.get('remaining_time'),
                obstacles=self.obstacles,
                map_size=self.map_size
            )
            self.tasks.append(task)
            # Add task to corresponding induction station
            station = self.induction_stations.get(data['start_point'])
            if station:
                station.add_task(task)
            else:
                print(f"Warning: Cannot find induction station {data['start_point']}")
        print("Task data loaded")

    def generate_a_star_with_turn_cost_map(self):
        """Generate A* path cost dictionary with turn cost between any two points in current map"""
        self.map_path_cost = {}
        for x1 in range(1, self.map_size[0] + 1):
            for y1 in range(1, self.map_size[1] + 1):
                for start_pitch in [0, 90, 180, 270]:
                    for x2 in range(1, self.map_size[0] + 1):
                        for y2 in range(1, self.map_size[1] + 1):
                            if (x1, y1) in self.obstacles or (x2, y2) in self.obstacles:
                                continue
                            cost = PathUtils.a_star_with_turn(
                                start=(x1, y1),
                                goal=(x2, y2),
                                start_pitch=start_pitch,
                                obstacles=self.obstacles,
                                map_size=self.map_size
                            )[1]
                            self.map_path_cost[str(((x1, y1, start_pitch), (x2, y2)))] = cost

    def save_to_json(self, filename: str) -> bool:
        """Save dictionary to JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.map_path_cost, f, indent=4, ensure_ascii=False)
            print(f"数据已保存到: {filename}")
            return True
        except Exception as e:
            print(f"保存失败: {e}")
            return False
    
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
    
    def run(self):
        """Run planner until all tasks complete"""
        try:
            print("Starting planner")
            start_time = time.time()

            while not self.all_tasks_complete():
                self.run_one_timestamp()

                # Check if planner runtime has exceeded timeout
                current_planning_time = time.time() - start_time
                if current_planning_time > self.max_planning_time:
                    raise KeyboardInterrupt(f"Planner runtime exceeded {self.max_planning_time} seconds, aborted")

            end_time = time.time()
            print(f"Planner execution ended, runtime: {end_time - start_time:.2f} seconds")
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
                x['name'].lower()  # Secondary sort key: Name first letter (case insensitive)
            )
        )
        with open(DataLoader.get_trajectory_path(), 'w', newline='') as csvfile:
            writer = DictWriter(csvfile, fieldnames=fieldnames)
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
        if result['valid_completed_num'] == len(self.tasks):
            print("All tasks completed correctly")
        else:
            print(f"Correctly completed tasks: {result['valid_completed_num']} / {len(self.tasks)}")
            print(f"False pickup count: {result['false_pickup_count']}")
            for error in result['false_pickups']:
                print(f"Time {error['timestamp']} sec, AGV {error['agv']} false pickup, details: {error['details']}")
        print(f"Total completion time: {result['total_completion_time']} seconds")
        service_time = 0
        for t in self.completed_tasks:
            service_time  += t.end_timestamp - t.available_timestamp
        print(f"Total completed tasks: {len(self.completed_tasks)}")
        print(f"Average service time: {service_time / len(self.completed_tasks):.2f} seconds")
        print(f"Collision count: {result['collision_count']}")
        if result['collision_count'] > 0:
            print("Collision details:")
            for collision in result['collisions']:
                print(f"Time {collision['timestamp']} sec, position: ({collision['X']}, {collision['Y']}), type: {collision['type']}, involved AGV list: {collision['AGVs']}")
        
        if not result['trajectory_valid']:
            print("\nTrajectory error details:")
            for error in result['trajectory_errors']:
                print(f"[{error['type']}] {error['agv']} at {error['timestamp']} sec: {error['details']}")

    def all_tasks_complete(self) -> bool:
        """Check if all induction stations have completed all tasks"""
        for induction in self.induction_stations.values():
            if not induction.is_complete():
                return False
        for t in self.tasks:
            if not t.is_completed():
                return False
        return True

    def run_one_timestamp(self):
        """Run one timestep"""

        allocator = BSA(
            agents=list(self.agents.values()),
            induction_stations=list(self.induction_stations.values()),
            obstacles=self.obstacles,
            map_size=self.map_size,
            map_cost=self.map_path_cost
        )
        allocate_results: List[Dict[str, any]] = allocator.allocate_tasks()
        
        # Generate trajectory to pickup point for AGVs assigned tasks
        if allocate_results:
            for result in allocate_results:
                agent_id = result['agent_id']
                task: Task = result['task']
                agent = self.agents.get(agent_id)
                agent.assign_task(task)
                self.plan_path_for_agent(agent_id)
        allocated_agent_ids = [result['agent_id'] for result in allocate_results] if allocate_results else []
        need_clear_assign_info_agent_ids = [agent.id for agent in self.agents.values() if agent.status == AgentStatus.MOVING_TO_PICKUP and agent.id not in allocated_agent_ids]
        self.clear_reservation_table(need_clear_assign_info_agent_ids, self.current_timestamp)
        for agent_id in need_clear_assign_info_agent_ids:
            agent = self.agents.get(agent_id)
            agent.clear_assign_task()

        # Plan path to nearest map edge for AGVs that are still idle
        for agent in self.agents.values():
            if agent.is_idle() and not agent.at_edge():
                obstacles = self.obstacles.copy()
                for ag in self.agents.values():
                    if agent.id == ag.id:
                        continue
                    if ag.is_idle():
                        obstacles.append(ag.get_position())
                    elif ag.is_moving_to_idle():
                        obstacles.append(ag.get_destination()[:2])
                nearest_edge = self.find_nearest_edge(agent.get_position(), agent.pitch, obstacles)
                agent.set_destination(nearest_edge + (None,))
                agent.set_status(AgentStatus.MOVING_TO_IDLE)
                self.plan_path_for_agent(agent.id)

        # Check if next AGV action will cause conflict and resolve conflicts
        while True:
            has_conflict, conflict_details = self.next_action_have_conflict()
            if not has_conflict:
                break
            print(f"Timestep {self.current_timestamp} conflict detected:")
            for conf in conflict_details:
                print(f" - {conf['description']}")
        # # Replace CAR with GCBS
        #     obstacles = self.obstacles.copy()
        #     for agent in self.agents.values():
        #         if agent.is_idle():
        #             obstacles.append(agent.get_position())

        #     agent_info = {}
        #     for agent_id in self.agents.keys():
        #         agent = self.agents[agent_id]
        #         if agent.is_idle():
        #             continue
        #         agent_info[agent_id] = {
        #             'start': agent.get_position(),
        #             'goal': agent.get_destination()[:2],
        #             'start_pitch': agent.pitch,
        #             'end_pitch': agent.get_destination()[2],
        #             'status': agent.status
        #         }
        #     self.clear_reservation_table(list(agent_info.keys()), self.current_timestamp)

        #     planner = CBSVariantsSolver(conflict_agvs=agent_info,
        #                                 conflict_timestamp=self.current_timestamp,
        #                                 reservation_table={},
        #                                 obstacles=obstacles,
        #                                 map_size=self.map_size,
        #                                 cost_map=self.map_path_cost)
            
        #     paths, _ = planner.find_solution()
        #     for agent_id, path in paths.items():
        #         agent = self.agents.get(agent_id)
        #         if agent.status in [AgentStatus.MOVING_TO_PICKUP, AgentStatus.MOVING_TO_DELIVERY]:
        #             path = path[1:-1]
        #         else:
        #             path = path[1:]
        #         agent.set_path(path)
        #         if len(path) > 0:
        #             goal = path[-1][:2]
        #         else:
        #             goal = agent.get_position()
        #         self.reserve_path(agent_id, path, self.current_timestamp, agent.status, goal)
            
            conflict_clusters = self.form_conflict_clusters(conflict_details)
            for cluster in conflict_clusters:
                print(f"Replanning path for conflict cluster, involved AGVs: {cluster}")
                solver = CBSVariantsSolver(
                    conflict_agvs=self.get_conflict_agents_info(cluster),
                    conflict_timestamp=self.current_timestamp,
                    reservation_table=self.reservation_table,
                    obstacles=self.get_obstacles_for_cbs(cluster),
                    map_size=self.map_size,
                    cost_map=self.map_path_cost
                )
                new_paths, have_solution = solver.find_solution()
                if new_paths is None:
                    solver = CBSVariantsSolver(
                        conflict_agvs=self.get_conflict_agents_info(cluster),
                        conflict_timestamp=self.current_timestamp,
                        reservation_table={},
                        obstacles=self.get_obstacles_for_cbs(cluster),
                        map_size=self.map_size,
                        cost_map=self.map_path_cost
                    )
                    new_paths, have_solution = solver.find_solution()
                    have_solution = False
                for agent_id in new_paths.keys():
                    agent = self.agents.get(agent_id)
                    if agent.status in [AgentStatus.MOVING_TO_PICKUP, AgentStatus.MOVING_TO_DELIVERY]:
                        path = new_paths[agent_id][1:-1]
                    else:
                        path = new_paths[agent_id][1:]
                    agent.set_path(path)
                    if have_solution:
                        if len(path) > 0:
                            goal = path[-1][:2]
                        else:
                            goal = agent.get_position()
                        self.reserve_path(agent_id, path, self.current_timestamp, agent.status, goal)
                        

        # Update timestep
        self.current_timestamp += 1
        for station in self.induction_stations.values():
            station.update_timestamp(self.current_timestamp)
        for agent in self.agents.values():
            agent.update_timestamp(self.current_timestamp)
        print(f"Timestep advanced to {self.current_timestamp}")

        # All AGVs execute actions
        for agent in self.agents.values():
            agent.generate_next_action()
            self.execute_agent_action(agent)
            agent.record_trajectory()
        
        # print('------------------------------')

    def execute_agent_action(self, agent: Agent):
        """Execute AGV current action"""
        if agent.get_action() in [AgentActions.MOVE, AgentActions.TURN]:
            next_x, next_y, next_pitch = agent.path.pop(0)
            agent.x = next_x
            agent.y = next_y
            agent.pitch = next_pitch
        elif agent.get_action() == AgentActions.LOAD:
            agent.load_task()
            self.plan_path_for_agent(agent.id)
            induction = self.induction_stations.get(agent.get_current_task().get_induction_station_id())
            cur_task = induction.load_next_task()
            if cur_task is None:
                print(f"Error: Induction station {induction.id} has no tasks to pickup, current Agent: {agent.id} task: {agent.get_current_task().id}")
            if cur_task.id != agent.get_current_task().id:
                print(f"Error: AGV {agent.id} picked wrong task {cur_task.id} != {agent.get_current_task().id}")
        elif agent.get_action() == AgentActions.UNLOAD:
            complete_task = agent.unload_task()
            induction = self.induction_stations.get(complete_task.get_induction_station_id())
            induction.task_completed(complete_task)
            self.completed_tasks.append(complete_task)
            if not agent.at_edge():
                obstacles = self.obstacles.copy()
                for ag in self.agents.values():
                    if agent.id == ag.id:
                        continue
                    if ag.is_idle():
                        obstacles.append(ag.get_position())
                    elif ag.is_moving_to_idle():
                        obstacles.append(ag.get_destination()[:2])
                nearest_edge = self.find_nearest_edge(agent.get_position(), agent.pitch, obstacles)
                agent.set_destination(nearest_edge + (None,))
                agent.set_status(AgentStatus.MOVING_TO_IDLE)
                self.plan_path_for_agent(agent.id)
        elif agent.get_action() == AgentActions.WAIT:
            if len(agent.path) > 0 and (agent.x, agent.y, agent.pitch) == agent.path[0]:
                agent.path.pop(0)
            elif len(agent.path) == 0:
                agent.set_status(AgentStatus.IDLE)

    def plan_path_for_agent(self, agent_id: str):
        agent = self.agents.get(agent_id)
        destination = agent.get_destination()
        obstacles = self.obstacles.copy()
        for ag in self.agents.values():
            if ag.id != agent.id and ag.is_idle():
                obstacles.append(ag.get_position())
        self.clear_reservation_table([agent_id], self.current_timestamp)
        path = self.space_time_astar(
            agent_id=agent_id,
            start=agent.get_position(),
            goal=destination[:2],
            start_timestamp=self.current_timestamp,
            start_pitch=agent.pitch,
            obstacles=obstacles,
            end_pitch=destination[2],
            status=agent.status
        )
        if path is None: # 时空A*无法规划出路径
            path = PathUtils.a_star_with_turn(
                start=agent.get_position(),
                goal=destination[:2],
                start_pitch=agent.pitch,
                obstacles=obstacles,
                map_size=self.map_size,
                end_pitch=destination[2]
            )[0]
        agent.set_path(path)

    def space_time_astar(self, 
                         agent_id:str, 
                         start:Tuple[int, int], 
                         goal:Tuple[int, int], 
                         start_timestamp:int, 
                         start_pitch:int, 
                         obstacles:List[Tuple[int, int]],
                         end_pitch:Optional[int]=None, 
                         status:Optional[AgentStatus]=None) -> Optional[List[Tuple[int, int, int]]]:
        """Space-time A* path planning algorithm"""
        directions = {
            'right': (1, 0),    # Right
            'up': (0, 1),    # Up
            'left': (-1, 0),  # Left
            'down': (0, -1),   # Down
            'stay': (0, 0)  # Stay in place
        }
        move_angles = {
            'right': 0,
            'up': 90,
            'left': 180,
            'down': 270
        }
        start_state = (start[0], start[1], start_pitch, 0)
        frontier: List[Tuple[int, int, int, Tuple[int, int, int, int], List[Tuple[int, int, int]]]] = []
        g0 = 0  # Time already spent to reach current state
        h0 = self.map_path_cost[str(((start[0], start[1], start_pitch), (goal[0], goal[1])))]
        heapq.heappush(frontier, (g0 + h0, h0, g0, start_state, []))
        visited = {}
        visited[start_state] = g0 + h0
        while frontier:
            f, h, g, state, path = heapq.heappop(frontier)
            if len(path) > 2 * (self.map_size[0] + self.map_size[1] + 1):  # Exceeds maximum possible path length, abort
                print(f"Path length exceeds maximum, exiting path planning!")
                break
            x, y, pitch, t = state
            if (x, y) == (goal[0], goal[1]):
                if status in [AgentStatus.MOVING_TO_PICKUP, AgentStatus.MOVING_TO_DELIVERY]:
                    if self.have_vertex_conflict(x, y, g + PathUtils.LOAD_UNLOAD_TIME + start_timestamp, agent_id):
                        continue
                if end_pitch is not None and pitch != end_pitch:
                    if self.have_vertex_conflict(x, y, g + PathUtils.TURN_COST + start_timestamp, agent_id):
                        continue
                    path.append((x, y, end_pitch))
                self.reserve_path(agent_id, path, start_timestamp, status, goal)
                return path
            for action in ['right', 'up', 'left', 'down', 'stay']:
                dx, dy = directions[action]
                new_x, new_y = x + dx, y + dy
                new_pitch = pitch if action == 'stay' else move_angles[action]
                if (new_x < 1 or new_x > self.map_size[0] or 
                    new_y < 1 or new_y > self.map_size[1] or 
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
        """Check if vertex conflict exists"""
        if t in self.reservation_table and (x, y) in self.reservation_table[t]:
            if agent_id is None or self.reservation_table[t][(x, y)] != agent_id:
                return True
        idle_agents_position = [agent.get_position() for agent in self.agents.values() if agent.is_idle()]
        if (x, y) in idle_agents_position:
            return True
        return False

    def have_swap_conflict(self, 
                            x: int, 
                            y: int, 
                            t: int, 
                            new_x: int, 
                            new_y: int):
        """Check if swap conflict exists"""
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
                      start_timestamp: int, 
                      status: AgentStatus,
                      goal: Tuple[int, int]):
        """Reserve path for Agent in reservation table"""
        for t, (x, y, _) in enumerate(path, start=start_timestamp + 1):
            self.reservation_table.setdefault(t, {})[(x, y)] = agent_id
        if status in [AgentStatus.MOVING_TO_PICKUP, AgentStatus.MOVING_TO_DELIVERY]:
            load_timestamp = start_timestamp + len(path) + PathUtils.LOAD_UNLOAD_TIME
            self.reservation_table.setdefault(load_timestamp, {})[goal] = agent_id

    def clear_reservation_table(self, 
                                agent_ids: List[str], 
                                timestep: int):
        """Clear paths after specified timestep in reservation table for all Agents in list"""
        keys_to_delete = [t for t in self.reservation_table if t > timestep]
        for t in keys_to_delete:
            for (x, y) in list(self.reservation_table[t].keys()):
                if self.reservation_table[t][(x, y)] in agent_ids:
                    del self.reservation_table[t][(x, y)]
    
    def print_reservation_table_after_timestamp(self, timestamp: int):
        """Print reservation table content after specified timestep"""
        print(f"Reservation table content after timestep {timestamp}:")
        for t in sorted(self.reservation_table.keys()):
            if t >= timestamp:
                print(f" Timestep {t}:")
                for (x, y), agent_id in self.reservation_table[t].items():
                    print(f"   Position ({x}, {y}): AGV {agent_id}")

    def find_nearest_edge(self, position: Tuple[int, int], pitch: int, obstacles: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Find nearest map edge point from current position
        """
        edge_points = []
        for i in range(2, self.map_size[0]):
            edge_points.append((i, 1))  # Bottom edge
            edge_points.append((i, self.map_size[1]))  # Top edge
        for j in range(2, self.map_size[1]):
            edge_points.append((1, j))  # Left edge
            edge_points.append((self.map_size[0], j))  # Right edge
        if self.map_size[0] == 70 and self.map_size[1] == 43:
            edge_points = list(set(edge_points + self.agents_initial_positions))
        min_distance = float('inf')
        nearest_edge = None
        for edge in edge_points:
            if edge in obstacles:
                continue
            distance = self.map_path_cost[str(((position[0], position[1], pitch), (edge[0], edge[1])))]
            if distance < min_distance:
                min_distance = distance
                nearest_edge = edge
        return nearest_edge

    def next_action_have_conflict(self) -> Tuple[bool, List[Dict]]:
        """
        Detect if next action will cause conflict
        :return: (whether conflict occurs, list of conflict details)
        """
        conflicts = []
        has_conflict = False
        
        # Detect vertex collision
        position_agent_map: Dict[Tuple[int, int], List[str]] = {}  # Map position to AGV list
        for agent_id, agent in self.agents.items():
            next_pos = agent.get_next_position()
            if next_pos not in position_agent_map:
                position_agent_map[next_pos] = []
            position_agent_map[next_pos].append(agent_id)
        for next_pos, agent_ids in position_agent_map.items():
            if len(agent_ids) > 1:
                has_conflict = True
                conf = {
                    'type': 'vertex',
                    'agents': agent_ids,
                    'description': f"Vertex collision: {', '.join(agent_ids)} will collide at position {next_pos}"
                }
                conflicts.append(conf)
        
        # Detect swap collision (edge collision)
        for agent1_id, agent1 in self.agents.items():
            current_pos1 = agent1.get_position()
            next_pos1 = agent1.get_next_position()
            for agent2_id, agent2 in self.agents.items():
                if agent1_id == agent2_id:
                    continue
                current_pos2 = agent2.get_position()
                next_pos2 = agent2.get_next_position()
                if next_pos1 == current_pos2 and next_pos2 == current_pos1:
                    if (abs(current_pos1[0] - current_pos2[0]) + 
                        abs(current_pos1[1] - current_pos2[1])) == 1:
                        has_conflict = True
                        has_recorded = any(conf['type'] == 'edge' and set(conf['agents']) == set([agent1_id, agent2_id]) for conf in conflicts)
                        if not has_recorded:
                            conf = {
                                'type': 'edge',
                                'agents': [agent1_id, agent2_id],
                                'description': f"Swap collision: {agent1_id} from {current_pos1} to {next_pos1} swaps position with AGV {agent2_id} from {current_pos2} to {next_pos2}"
                            }
                            conflicts.append(conf)
        
        return has_conflict, conflicts

    def form_conflict_clusters(self, collisions: List[Dict[str, Any]]) -> List[List[str]]:
        """
        Form conflict clusters based on collision information        
        Args:
            collisions: List of collision information, each element is a dictionary {
                'type': collision type,
                'agents': list of involved AGVs,
                'description': description string
            }
        Returns:
            clusters: List of conflict clusters, each cluster is a set of AGVs
        """
        # Initial clustering - merge collisions with common AGVs
        clusters = []
        for collision in collisions:
            collision_set = set(collision['agents'])
            merged_cluster = set(collision_set)
            new_clusters = []
            for cluster in clusters:
                if cluster & collision_set:  # If there is intersection
                    merged_cluster |= cluster  # Merge cluster
                else:
                    new_clusters.append(cluster)  # Keep independent cluster
            new_clusters.append(merged_cluster)
            clusters = new_clusters
        
        # Expand clusters to include neighboring AGVs
        pos_to_agv = {}  # Create position to AGV mapping
        agv_positions = {agv.id: agv.get_position() for agv in self.agents.values()}
        for agv_id, pos in agv_positions.items():
            pos_to_agv[pos] = agv_id
        assigned_agvs = set()  # Track which AGVs have been assigned to clusters
        for cluster in clusters:
            assigned_agvs |= cluster
        
        # Use BFS to expand each cluster
        expanded_clusters = []
        for cluster in clusters:
            queue = deque(cluster)
            expanded_cluster = set(cluster)
            while queue:
                current_agv = queue.popleft()
                current_pos = agv_positions[current_agv]
                directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                for dx, dy in directions:
                    neighbor_pos = (current_pos[0] + dx, current_pos[1] + dy)
                    if neighbor_pos in pos_to_agv:
                        neighbor_agv = pos_to_agv[neighbor_pos]
                        if neighbor_agv in expanded_cluster or neighbor_agv in assigned_agvs:
                            continue
                        expanded_cluster.add(neighbor_agv)
                        queue.append(neighbor_agv)
                        assigned_agvs.add(neighbor_agv)
            expanded_clusters.append(expanded_cluster)
        
        # Merge connected clusters
        final_clusters = []
        while expanded_clusters:
            current_cluster = expanded_clusters.pop()
            merged = False
            for i, other_cluster in enumerate(final_clusters): # Check if there are connected clusters
                for agv1 in current_cluster: # Check if there are neighboring AGVs between clusters (connected by position)
                    pos1 = agv_positions[agv1]
                    for agv2 in other_cluster:
                        pos2 = agv_positions[agv2]
                        if (abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])) == 1:
                            final_clusters[i] |= current_cluster
                            merged = True
                            break
                    if merged:
                        break
                if merged:
                    break
            if not merged:
                final_clusters.append(current_cluster)

        return [sorted(list(cluster), key=lambda x: int(x[3:])) for cluster in final_clusters]
    
    def get_conflict_agents_info(self, agent_ids: List[str]) -> Dict[str, Any]:
        """Get detailed information of conflicting AGVs"""
        info = {}
        for agent_id in agent_ids:
            agent = self.agents.get(agent_id)
            if agent.is_idle() or agent.get_position() == agent.get_destination()[:2] and agent.status == AgentStatus.MOVING_TO_IDLE:
                continue
            info[agent_id] = {
                'start': agent.get_position(),
                'goal': agent.get_destination()[:2],
                'start_pitch': agent.pitch,
                'end_pitch': agent.get_destination()[2],
                'status': agent.status
            }
        self.clear_reservation_table(list(info.keys()), self.current_timestamp)
        return info
    
    def get_obstacles_for_cbs(self, agent_ids: List[str]) -> List[Tuple[int, int]]:
        """Get obstacle list for CBS use"""
        obstacles = self.obstacles.copy()
        for agent in self.agents.values():
            if agent.is_idle():
                obstacles.append(agent.get_position())
        for id in agent_ids:
            agent = self.agents.get(id)
            if agent.status == AgentStatus.MOVING_TO_IDLE and agent.get_position() == agent.get_destination()[:2] and agent.get_position() not in obstacles:
                obstacles.append(agent.get_position())
        return obstacles

if __name__ == "__main__":
    planner = Planner()
    planner.run()