import heapq
from typing import List, Tuple, Optional

class PathUtils:
    TURN_COST = 1     # Turn cost
    LOAD_UNLOAD_TIME = 1 # Load/unload time

    @staticmethod
    def manhattan(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance
        """
        if not (len(p1) == 2 and len(p2) == 2):
            raise ValueError("Parameter for calculating Manhattan distance is incorrect, the length of p1 and p2 must be 2.")
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
    @staticmethod
    def get_orientation(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> int:
        """
        Calculate the angle for turning from from_pos to to_pos
        """
        if not (len(from_pos) == 2 and len(to_pos) == 2):
            raise ValueError("Parameter for calculating orientation is incorrect, the length of from_pos and to_pos must be 2.")
        dx, dy = to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]
        if dx > 0:
            return 0
        elif dx < 0:
            return 180
        elif dy > 0:
            return 90
        elif dy < 0:
            return 270
        else:
            return None
        
    @staticmethod
    def a_star_with_turn(start: Tuple[int, int], 
                         goal: Tuple[int, int], 
                         start_pitch: int, 
                         obstacles: List[Tuple[int, int]], 
                         map_size: Tuple[int, int],
                         end_pitch: Optional[int] = None) -> Tuple[List[Tuple[int, int, int]], float]:
        """
        A* algorithm path planning (considering turn cost)
        Used to plan ideal path to pickup point for AGV during bidding (without considering inter-AGV collisions)
        """
        angle_to_vec = {
            0: (1, 0),    # X positive direction
            90: (0, 1),    # Y positive direction
            180: (-1, 0),  # X negative direction
            270: (0, -1)   # Y negative direction
        }
        start_state = (start[0], start[1], start_pitch) # Start state: (x, y, current orientation angle)
        
        # Priority queue: (f_score, h_score, g_score, state, path)
        frontier: List[Tuple[int, int, int, Tuple[int, int, int], List[Tuple[int, int, int]]]] = []
        g0 = 0
        h0 = PathUtils.manhattan(start, goal)
        heapq.heappush(frontier, (g0 + h0, h0, g0, start_state, []))

        # Record visited states and their minimum cost
        visited = {}
        visited[start_state] = g0
        while frontier:
            f, h, g, state, path = heapq.heappop(frontier)
            x, y, current_angle = state
            if (x, y) == (goal[0], goal[1]):
                if end_pitch is not None and current_angle != end_pitch:
                    path.append((x, y, end_pitch))
                    f += PathUtils.TURN_COST
                return path, f
            for move_angle in [0, 90, 180, 270]:
                dx, dy = angle_to_vec[move_angle]
                nx, ny = x + dx, y + dy
                if (nx < 1 or nx > map_size[0] or 
                    ny < 1 or ny > map_size[1] or 
                    (nx, ny) in obstacles):
                    continue
                if move_angle == current_angle:
                    move_cost = 1  # Same direction, only move time needed
                    new_path = path + [(nx, ny, move_angle)]
                else:
                    move_cost = 1 + PathUtils.TURN_COST  # Different direction, turn + move time
                    new_path = path + [(x, y, move_angle), (nx, ny, move_angle)]
                new_g = g + move_cost
                new_state = (nx, ny, move_angle)
                if new_state not in visited or visited[new_state] > new_g:
                    visited[new_state] = new_g
                    new_h = PathUtils.manhattan((nx, ny), goal)
                    new_f = new_g + new_h
                    heapq.heappush(frontier, (new_f, new_h, new_g, new_state, new_path))

        return [], float('inf')  # No feasible path
    