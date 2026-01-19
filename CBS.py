import heapq
from typing import List, Tuple, Dict, Optional, Any
from PathUtils import PathUtils
from Agent import AgentStatus
import time

# Algorithm type enumeration
ALGORITHM_TYPES = {
    "CBS": "CBS",
    "GCBS_H": "GCBS_H",
    "GCBS_L": "GCBS_L", 
    "GCBS_HL": "GCBS_HL",
    "ECBS": "ECBS"
}

def detect_collision(path1, path2):
    """Return first collision between two paths or None (if no collision)"""
    def get_location(path, time):
        if time < 0:
            return (path[0][0], path[0][1])
        elif time < len(path):
            return (path[time][0], path[time][1])
        else:
            return None

    def vertex_collision(t):
        agv_1_curr_loc = get_location(path1, t)
        agv_2_curr_loc = get_location(path2, t)
        if agv_1_curr_loc is None or agv_2_curr_loc is None:
            return None
        if agv_1_curr_loc == agv_2_curr_loc:
            return {'loc': [agv_1_curr_loc], 'timestep': t}
        return None

    def edge_collision(t):
        agv_1_prev_loc = get_location(path1, t-1)
        agv_2_prev_loc = get_location(path2, t-1)
        agv_1_curr_loc = get_location(path1, t)
        agv_2_curr_loc = get_location(path2, t)
        if agv_1_prev_loc is None or agv_2_prev_loc is None or \
           agv_1_curr_loc is None or agv_2_curr_loc is None:
            return None
        if agv_1_prev_loc == agv_2_curr_loc and agv_2_prev_loc == agv_1_curr_loc:
            return {'loc': [agv_1_prev_loc, agv_1_curr_loc], 'timestep': t}
        return None

    if vertex_collision(t = 0) is not None:
        return vertex_collision(t = 0)
    for t in range(1, max(len(path1), len(path2))):
        if vertex_collision(t) is not None:
            return vertex_collision(t)
        if edge_collision(t) is not None:
            return edge_collision(t)

    return None

def detect_collisions(paths):
    """
    Detect collisions between all AGV paths
    """
    collisions = []
    agv_id_list = paths.keys()
    for i, agv_1 in enumerate(agv_id_list):
        for j, agv_2 in enumerate(agv_id_list):
            if i >= j:
                continue
            collision = detect_collision(paths[agv_1], paths[agv_2])
            if collision is not None:
                collisions.append({'agv_1': agv_1, 'agv_2': agv_2, \
                                   'loc': collision['loc'], \
                                   'timestep': collision['timestep']})
    return collisions


def standard_splitting(collision):
    """
    Split conflict, for given collision, return two constraints to resolve collision
    For vertex collision, return two constraints to prevent two AGVs from reaching collision position at collision time
    For edge collision, return two constraints to prevent two AGVs from traversing collision edge in original path order at collision time
    """
    VERTEX_COLLISION, EDGE_COLLISION = 0, 1

    def collision_type():
        if len(collision['loc']) == 1:
            return VERTEX_COLLISION
        return EDGE_COLLISION

    def constraint(agv_sequence, reverse_edge):
        agv_id = collision[f"agv_{agv_sequence}"]
        collision_loc = collision['loc'].copy()
        if reverse_edge:
            collision_loc.reverse()
        return {'agv_id': agv_id,
                'loc': collision_loc,
                'timestep': collision['timestep']}

    constraints = []
    if collision_type() == VERTEX_COLLISION:
        constraints.append(constraint(agv_sequence = 1, reverse_edge = False))
        constraints.append(constraint(agv_sequence = 2, reverse_edge = False))
    elif collision_type() == EDGE_COLLISION:
        constraints.append(constraint(agv_sequence = 1, reverse_edge = False))
        constraints.append(constraint(agv_sequence = 2, reverse_edge = True))

    return constraints

class CBSVariantsSolver:
    def __init__(self, 
                 conflict_agvs: Dict[str, Dict[str, Any]], 
                 conflict_timestamp: int, 
                 reservation_table: Dict[int, Dict[Tuple[int, int], str]], 
                 obstacles: List[Tuple[int, int]],
                 map_size: Tuple[int, int],
                 cost_map,
                 algorithm_type: str = "GCBS_HL",
                 suboptimality_bound: float = 1.1):
        """
        CBS algorithm variant solver
        
        Args:
            conflict_agvs: Conflict AGV information
            conflict_timestamp: Conflict timestamp
            reservation_table: Reservation table
            obstacles: Obstacle list
            map_size: Map size
            cost_map: Cost map
            algorithm_type: Algorithm type, options: "CBS", "GCBS_H", "GCBS_L", "GCBS_HL", "ECBS"
            suboptimality_bound: Suboptimality factor w (ECBS only)
        """
        self.conflict_agvs = conflict_agvs  # {"<agent_id>": {"start": <start>, "goal": <goal>, "start_pitch": <start_pitch>, "end_pitch": <end_pitch>, "status": <status>}}
        self.reservation_table = reservation_table
        self.obstacles = obstacles
        self.map_size = map_size
        self.conflict_timestamp = conflict_timestamp
        self.algorithm_type = algorithm_type
        self.suboptimality_bound = suboptimality_bound

        self.start_time = time.time()
        self.max_running_time = 300  # Max running time in seconds

        # Validate algorithm type
        if algorithm_type not in ALGORITHM_TYPES.values():
            raise ValueError(f"Unsupported algorithm type: {algorithm_type}, options: {list(ALGORITHM_TYPES.values())}")

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.open_list = []  # For standard CBS and GCBS
        self.focal_list = []  # For ECBS
        # ECBS-specific variables
        self.best_lb = float('inf')  # Best lower bound

        self.map_path_cost = cost_map

    def compute_conflict_heuristic(self, paths):
        """
        Compute conflict heuristic h_c (h3 in paper: number of conflict pairs)
        
        Args:
            paths: Path dictionary
            
        Returns:
            h_c value (number of conflict pairs)
        """
        if not paths:
            return 0
            
        agv_ids = list(paths.keys())
        conflict_pairs = 0
        
        for i in range(len(agv_ids)):
            for j in range(i+1, len(agv_ids)):
                agv1 = agv_ids[i]
                agv2 = agv_ids[j]
                if detect_collision(paths[agv1], paths[agv2]) is not None:
                    conflict_pairs += 1
                    
        return conflict_pairs
    
    def compute_conflict_heuristic_for_agent(self, agent_path, other_paths):
        """
        Compute conflict heuristic for a single AGV path against other paths
        
        Args:
            agent_path: Current AGV's path
            other_paths: Other AGVs' path dictionary
            
        Returns:
            h_c value (number of conflicting AGVs)
        """
        if not agent_path or not other_paths:
            return 0
            
        conflict_count = 0
        for agv_id, path in other_paths.items():
            if detect_collision(agent_path, path) is not None:
                conflict_count += 1
                
        return conflict_count

    def push_node(self, node, lb=None):
        """
        Push node into appropriate priority queue
        
        Args:
            node: CT node
            lb: Lower bound (ECBS only)
        """
        if self.algorithm_type == "ECBS":
            # ECBS使用focal search
            node_cost = node['cost']
            
            # Compute lower bound (if not provided)
            if lb is None:
                lb = self.compute_lower_bound(node)
                
            # Update global best lower bound
            if lb < self.best_lb:
                self.best_lb = lb
                
            # Check if should be added to focal list
            if node_cost <= self.best_lb * self.suboptimality_bound:
                # Compute conflict heuristic
                h_c = self.compute_conflict_heuristic(node['paths'])
                heapq.heappush(self.focal_list, (h_c, node_cost, len(node['collisions']), self.num_of_generated, node))
            
            # Always add to open list (to maintain lower bound)
            heapq.heappush(self.open_list, (node_cost, len(node['collisions']), self.num_of_generated, node))
            
        elif self.algorithm_type in ["GCBS_H", "GCBS_HL"]:
            # GCBS-H and GCBS-HL: high-level uses conflict heuristic
            h_c = self.compute_conflict_heuristic(node['paths'])
            heapq.heappush(self.open_list, (h_c, node['cost'], len(node['collisions']), self.num_of_generated, node))
            
        else:  # CBS and GCBS-L
            # Standard CBS and GCBS-L: high-level uses cost
            heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
            
        self.num_of_generated += 1

    def pop_node(self):
        """Pop node from priority queue"""
        if self.algorithm_type == "ECBS" and self.focal_list:
            # ECBS: pop from focal list first
            _, _, _, _, node = heapq.heappop(self.focal_list)
        else:
            # Other algorithms: pop from open list
            if self.algorithm_type in ["GCBS_H", "GCBS_HL"]:
                _, _, _, _, node = heapq.heappop(self.open_list)
            else:
                _, _, _, node = heapq.heappop(self.open_list)
            
        self.num_of_expanded += 1
        return node
    
    def compute_lower_bound(self, node):
        """
        Compute lower bound of node (for ECBS)
        
        Args:
            node: CT node
            
        Returns:
            Lower bound value
        """
        # Simplified implementation: use current cost as lower bound
        # In actual implementation should compute sum of optimal path costs for each agent (ignoring constraints)
        return node['cost']
    
    def get_path_cost(self, paths):
        """Calculate maximum path length as cost"""
        return max(len(path) for path in paths.values()) if paths else float('inf')

    def find_solution(self):
        """Find solution"""
        root = {'cost': 0,
                'constraints': [],
                'paths': {},
                'collisions': []}
        
        have_solution = True

        for agv_id in self.conflict_agvs:
            other_paths = {aid: root['paths'][aid] for aid in root['paths'] if aid != agv_id}
            path = self.a_star_for_cbs_variant(
                agv_id=agv_id,
                start=self.conflict_agvs[agv_id]['start'],
                goal=self.conflict_agvs[agv_id]['goal'],
                start_pitch=self.conflict_agvs[agv_id]['start_pitch'],
                end_pitch=self.conflict_agvs[agv_id]['end_pitch'],
                status=self.conflict_agvs[agv_id]['status'],
                constraints=root['constraints'],
                other_paths=other_paths  # Root node has no other paths
            ) # Path includes start point
            if path is None:
                have_solution = False
                self.reservation_table = {}
                break
            root['paths'][agv_id] = path

        if not have_solution:
            for agv_id in self.conflict_agvs:
                other_paths = {aid: root['paths'][aid] for aid in root['paths'] if aid != agv_id}
                path = self.a_star_for_cbs_variant(
                    agv_id=agv_id,
                    start=self.conflict_agvs[agv_id]['start'],
                    goal=self.conflict_agvs[agv_id]['goal'],
                    start_pitch=self.conflict_agvs[agv_id]['start_pitch'],
                    end_pitch=self.conflict_agvs[agv_id]['end_pitch'],
                    status=self.conflict_agvs[agv_id]['status'],
                    constraints=root['constraints'],
                    other_paths=other_paths  # 根节点没有其他路径
                )
                if path is None:
                    raise BaseException(f"AGV {agv_id} cannot find path even without constraints!")
                root['paths'][agv_id] = path

        root['cost'] = self.get_path_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])

        # Push root node into priority queue
        if self.algorithm_type == "ECBS":
            lb = self.compute_lower_bound(root)
            self.best_lb = lb
            self.push_node(root, lb)
        else:
            self.push_node(root)

        while self.open_list or (self.algorithm_type == "ECBS" and self.focal_list):
            
            if time.time() - self.start_time > self.max_running_time:
                raise KeyboardInterrupt(f"CBS algorithm runtime exceeded limit {self.max_running_time} seconds, aborted")

            # For ECBS, if focal list is empty but open list is not, need to update focal list
            if self.algorithm_type == "ECBS" and not self.focal_list and self.open_list:
                self.update_focal_list_for_high_level()
                if not self.focal_list:
                    break

            # if self.algorithm_type == "ECBS":
            #     print(f"扩展节点 #{self.num_of_expanded}, 生成数: {self.num_of_generated}, open_list大小: {len(self.open_list)}, focal_list大小: {len(self.focal_list)}")
            node = self.pop_node()
            # if self.algorithm_type == "ECBS":
            #     print(f"当前节点代价: {node['cost']}, 冲突数: {len(node['collisions'])}, 最优下界: {self.best_lb}")
            collisions = node['collisions']

            if not collisions:
                return node['paths'], have_solution

            first_collision = collisions[0]
            constraints = standard_splitting(first_collision)

            for constraint in constraints:
                child = {'cost': 0,
                        'constraints': node['constraints'].copy(),
                        'paths': node['paths'].copy(),
                        'collisions': []}
                child['constraints'].append(constraint)

                agv_id = constraint['agv_id']

                # Prepare other AGVs' paths (for conflict heuristic computation)
                other_paths = {aid: path for aid, path in child['paths'].items() if aid != agv_id}

                new_path = self.a_star_for_cbs_variant(
                    agv_id=agv_id,
                    start=self.conflict_agvs[agv_id]['start'],
                    goal=self.conflict_agvs[agv_id]['goal'],
                    start_pitch=self.conflict_agvs[agv_id]['start_pitch'],
                    end_pitch=self.conflict_agvs[agv_id]['end_pitch'],
                    status=self.conflict_agvs[agv_id]['status'],
                    constraints=child['constraints'],
                    other_paths=other_paths
                )

                if new_path is not None:
                    child['paths'][agv_id] = new_path
                    child['collisions'] = detect_collisions(child['paths'])
                    child['cost'] = self.get_path_cost(child['paths'])
                    
                    if self.algorithm_type == "ECBS":
                        lb = self.compute_lower_bound(child)
                        self.push_node(child, lb)
                    else:
                        self.push_node(child)

        return None, False
    
    def update_focal_list_for_high_level(self):
        """Update ECBS focal list (based on current best lower bound)"""
        if not self.open_list or self.suboptimality_bound <= 1.0:
            return
            
        # Clear current focal list
        self.focal_list = []
        # Recompute best lower bound
        min_cost, num, n_o_g, best_node = heapq.heappop(self.open_list)
        self.best_lb = min_cost
        h_c = self.compute_conflict_heuristic(best_node['paths'])
        heapq.heappush(self.focal_list, (h_c, min_cost, num, n_o_g, best_node))

        # Rebuild focal list
        n = len(self.open_list)
        new_open_list = []
        for _ in range(n):
            cost, collision_num, n_o_g, node = heapq.heappop(self.open_list)
            # Check if should be added to focal list
            if cost <= self.best_lb * self.suboptimality_bound:
                h_c = self.compute_conflict_heuristic(node['paths'])
                heapq.heappush(self.focal_list, (h_c, cost, collision_num, n_o_g, node))
            else:
                heapq.heappush(new_open_list, (cost, collision_num, n_o_g, node))
        self.open_list = new_open_list

    def a_star_for_cbs_variant(self, 
                               agv_id: str, 
                               start: Tuple[int, int], 
                               goal: Tuple[int, int], 
                               start_pitch: int, 
                               end_pitch: Optional[int] = None, 
                               status: Optional[AgentStatus] = None, 
                               constraints: List[Dict[str, Any]] = [],
                               other_paths: Dict[str, List[Tuple[int, int, int]]] = None):
        """
        A* algorithm variant, supports different algorithm types
        
        Args:
            other_paths: Other AGVs' paths (for conflict heuristic computation)
        """
        # Determine whether to use conflict heuristic based on algorithm type
        use_conflict_heuristic = (self.algorithm_type in ["GCBS_L", "GCBS_HL", "ECBS"])
        constraint_table = self.build_constraint_table(constraints, agv_id)

        def _break_vertex_constraints(loc, next_t):
            """Check vertex constraints for stationary case"""
            if next_t in constraint_table:
                constraints = constraint_table[next_t]
                for constraint in constraints:
                    if constraint['type'] == 'vertex' and loc == constraint['loc'][0]:
                        return True
            return False
        
        def _break_vertex_and_edge_constraints(current_loc, next_loc, next_t):
            """Check vertex and edge constraints for movement case"""
            if _break_vertex_constraints(next_loc, next_t):
                return True
            if next_t in constraint_table:
                constraints = constraint_table[next_t]
                for constraint in constraints:
                    if constraint['type'] == 'edge' and current_loc == constraint['loc'][0] and next_loc == constraint['loc'][1]:
                        return True
            return False

        def _break_constraints(x, y, t, new_x, new_y, new_t, stay=False, turn=False):
            """Check if current AGV violates constraints"""
            current_loc = (x, y)
            next_loc = (new_x, new_y)
            if stay:
                if _break_vertex_constraints(next_loc, new_t):
                    return True
            else:
                if turn:
                    if _break_vertex_constraints(current_loc, t + PathUtils.TURN_COST):
                        return True
                    if _break_vertex_and_edge_constraints(current_loc, next_loc, new_t):
                        return True
                else:
                    if _break_vertex_and_edge_constraints(current_loc, next_loc, new_t):
                        return True
            return False
        
        def _update_focal_list_for_low_level(open_list, focal_list):
            """Update low-level ECBS focal list (based on current best lower bound)"""
            if not open_list or self.suboptimality_bound <= 1.0:
                return
            
            # Clear current focal list
            focal_list = []
            # Recompute best lower bound
            f, h, g, state, n_o_g, path = heapq.heappop(open_list)
            best_lb = f
            h_c = self.compute_conflict_heuristic_for_agent(path, other_paths) if other_paths else 0
            heapq.heappush(focal_list, (h_c, f, h, g, state, n_o_g, path))

            # Rebuild focal list
            n = len(open_list)
            new_open_list = []
            for _ in range(n):
                f, h, g, state, num_generated, path = heapq.heappop(open_list)
                # Check if should be added to focal list
                if f <= best_lb * self.suboptimality_bound:
                    h_c = self.compute_conflict_heuristic_for_agent(path, other_paths) if other_paths else 0
                    heapq.heappush(focal_list, (h_c, f, h, g, state, num_generated, path))
                else:
                    heapq.heappush(new_open_list, (f, h, g, state, num_generated, path))
            
            return new_open_list, focal_list, best_lb

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
        start_state = (start[0], start[1], start_pitch, 0) # Start state: (x, y, current orientation, current timestep)

        open_list = []
        if self.algorithm_type == "ECBS":
            focal_list = []

        g0 = 0
        h0 = self.map_path_cost[str(((start[0], start[1], start_pitch), (goal[0], goal[1])))]
        num_of_generated = 0

        best_lb = g0 + h0

        if use_conflict_heuristic:
            # Compute conflict heuristic for initial state
            initial_path = [(start[0], start[1], start_pitch)]
            h_c0 = self.compute_conflict_heuristic_for_agent(initial_path, other_paths) if other_paths else 0
            
            if self.algorithm_type == "ECBS":
                heapq.heappush(open_list, (g0 + h0, h0, g0, start_state, num_of_generated, initial_path))
                num_of_generated += 1
                heapq.heappush(focal_list, (h_c0, g0 + h0, h0, g0, start_state, num_of_generated, initial_path))
            else:
                heapq.heappush(open_list, (h_c0, g0 + h0, h0, g0, start_state, num_of_generated, initial_path))
        else:
            heapq.heappush(open_list, (g0 + h0, h0, g0, start_state, num_of_generated, [(start[0], start[1], start_pitch)]))
        num_of_generated += 1

        visited = {}
        visited[start_state] = (h_c0, g0 + h0) if self.algorithm_type in ["GCBS_L", "GCBS_HL"] else g0 + h0
        while open_list or (self.algorithm_type == "ECBS" and focal_list):
            
            if self.algorithm_type == "ECBS" and not focal_list and open_list:
                open_list, focal_list, best_lb = _update_focal_list_for_low_level(open_list, focal_list)
                if not focal_list:
                    break

            if use_conflict_heuristic:
                if self.algorithm_type == "ECBS":
                    h_c, f, h, g, state, _, path = heapq.heappop(focal_list)
                else:
                    h_c, f, h, g, state, _, path = heapq.heappop(open_list)
            else:
                f, h, g, state, _, path = heapq.heappop(open_list)

            x, y, pitch, _ = state
            if (x, y) == (goal[0], goal[1]):
                if status in [AgentStatus.MOVING_TO_PICKUP, AgentStatus.MOVING_TO_DELIVERY] or (end_pitch is not None and pitch != end_pitch):
                    if self._have_vertex_conflict(x, y, g + 1 + self.conflict_timestamp):
                        continue
                    if _break_vertex_constraints((x, y), g + 1):
                        continue
                    if status in [AgentStatus.MOVING_TO_PICKUP, AgentStatus.MOVING_TO_DELIVERY]:
                        path.append((x, y, pitch))
                    if end_pitch is not None and pitch != end_pitch:
                        path.append((x, y, end_pitch))
                return path
            for action in ['right', 'up', 'left', 'down', 'stay']:
                dx, dy = directions[action]
                new_x, new_y, new_pitch = x + dx, y + dy, pitch
                if self._out_of_bounds(new_x, new_y) or self._have_obstacle(new_x, new_y):
                    continue
                stay, turn = False, False
                if action in ['right', 'up', 'left', 'down']:
                    new_pitch = move_angles[action]
                    if new_pitch == pitch:
                        new_g = g + 1
                        new_path = path + [(new_x, new_y, new_pitch)]
                    else:
                        new_g = g + PathUtils.TURN_COST + 1
                        new_path = path + [(x, y, new_pitch), (new_x, new_y, new_pitch)]
                        turn = True
                else:  # stay
                    new_g = g + 1
                    new_path = path + [(new_x, new_y, new_pitch)]
                    stay = True
                if self._have_conflict_with_reservation_table(x, y, g, new_x, new_y, new_g, stay=stay, turn=turn):
                    continue
                if _break_constraints(x, y, g, new_x, new_y, new_g, stay=stay, turn=turn):
                    continue
                new_h = self.map_path_cost[str(((new_x, new_y, new_pitch), (goal[0], goal[1])))]
                new_f = new_g + new_h
                new_state = (new_x, new_y, new_pitch, new_g)
                if use_conflict_heuristic:
                    # 计算新路径的冲突启发式
                    h_c_new = self.compute_conflict_heuristic_for_agent(new_path, other_paths) if other_paths else 0
                    if self.algorithm_type == "ECBS":
                        if new_state not in visited or visited[new_state] > new_f:
                            visited[new_state] = new_f
                            heapq.heappush(open_list, (new_f, new_h, new_g, new_state, num_of_generated, new_path))
                            num_of_generated += 1
                            if new_f <= self.suboptimality_bound * best_lb:
                                heapq.heappush(focal_list, (h_c_new, new_f, new_h, new_g, new_state, num_of_generated, new_path))
                                num_of_generated += 1
                            if new_f < best_lb:
                                best_lb = new_f
                    else:
                        if new_state not in visited or visited[new_state][0] > h_c_new or (visited[new_state][0] == h_c_new and visited[new_state][1] > new_f):
                            visited[new_state] = (h_c_new, new_f)
                            heapq.heappush(open_list, (h_c_new, new_f, new_h, new_g, new_state, num_of_generated, new_path))
                            num_of_generated += 1
                else:
                    if new_state not in visited or visited[new_state] > new_f:
                        visited[new_state] = new_f
                        heapq.heappush(open_list, (new_f, new_h, new_g, new_state, num_of_generated, new_path))
                        num_of_generated += 1
        return None
    
    def _have_conflict_with_reservation_table(self, x, y, t, new_x, new_y, new_t, stay=False, turn=False):
        """Check if position conflicts with reservation table"""
        if stay:
            if self._have_vertex_conflict(new_x, new_y, new_t + self.conflict_timestamp):
                return True
        else:
            if turn:
                if self._have_vertex_conflict(x, y, t + PathUtils.TURN_COST + self.conflict_timestamp):
                    return True
                if self._have_vertex_conflict(new_x, new_y, new_t + self.conflict_timestamp):
                    return True
                if self._have_swap_conflict(x, y, t + PathUtils.TURN_COST + self.conflict_timestamp, new_x, new_y, new_t + self.conflict_timestamp):
                    return True
            else:
                if self._have_vertex_conflict(new_x, new_y, new_t + self.conflict_timestamp):
                    return True
                if self._have_swap_conflict(x, y, t + self.conflict_timestamp, new_x, new_y, new_t + self.conflict_timestamp):
                    return True
        return False

    def _have_vertex_conflict(self, x, y, t):
        """Check if spacetime position has conflict"""
        return t in self.reservation_table and (x, y) in self.reservation_table[t]
    
    def _have_swap_conflict(self, x, y, t, new_x, new_y, new_t):
        """Check swap conflict"""
        return t in self.reservation_table and (new_x, new_y) in self.reservation_table[t] and \
               new_t in self.reservation_table and (x, y) in self.reservation_table[new_t] and \
               self.reservation_table[t][(new_x, new_y)] == self.reservation_table[new_t][(x, y)]
    
    def _out_of_bounds(self, x, y):
        """Check if position is out of bounds"""
        return x < 1 or x > self.map_size[0] or y < 1 or y > self.map_size[1]
    
    def _have_obstacle(self, x, y):
        """Check if position has obstacle"""
        return (x, y) in self.obstacles
    
    def build_constraint_table(self, constraints, agv_id):
        """Build constraint table that <agv_id> needs to satisfy"""
        constraint_table = {}
        
        def add_to_constraint_table(timestep, constraint_loc, constraint_type):
            constraint = {'loc': constraint_loc, 'type': constraint_type}

            if timestep in constraint_table:
                constraint_table[timestep].append(constraint)
            else:
                constraint_table[timestep] = []
                constraint_table[timestep].append(constraint)

        for constraint in constraints:
            if constraint['agv_id'] == agv_id:
                timestep = constraint['timestep']
                constraint_loc = constraint['loc']
                if 'type' in constraint:
                    constraint_type = constraint['type']
                else:
                    num_locs = len(constraint['loc'])
                    constraint_type = 'vertex' if num_locs == 1 else 'edge' if num_locs == 2 else None
                add_to_constraint_table(timestep, constraint_loc, constraint_type)

        return constraint_table