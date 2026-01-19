import pandas as pd
from collections import defaultdict, deque

def check_agv_trajectory(df: pd.DataFrame):
    """
    Check validity of AGV trajectory
    :param df: Trajectory data DataFrame
    :return: (all_valid, errors) - Whether all valid and error list
    """
    errors = []
    agv_groups = df.groupby('name')
    
    for name, group in agv_groups:
        group = group.sort_values('timestamp')
        prev_row = None
        
        for _, row in group.iterrows():
            if prev_row is not None:
                # Check state changes
                dx = row['X'] - prev_row['X']
                dy = row['Y'] - prev_row['Y']
                dp = (row['pitch'] - prev_row['pitch']) % 360
                loaded_changed = (row['loaded'] != prev_row['loaded'])

                # Movement check
                if abs(dx) > 0 or abs(dy) > 0:
                    if not ((dx == 1 and dy == 0 and prev_row['pitch'] == 0) or
                            (dx == 0 and dy == 1 and prev_row['pitch'] == 90) or
                            (dx == -1 and dy == 0 and prev_row['pitch'] == 180) or
                            (dx == 0 and dy == -1 and prev_row['pitch'] == 270)):
                        errors.append({
                            'type': 'Illegal movement',
                            'agv': name,
                            'timestamp': row['timestamp'],
                            'details': f"Moving from ({prev_row['X']},{prev_row['Y']}) at angle {prev_row['pitch']} to ({row['X']},{row['Y']})"
                        })
                    if row['pitch'] != prev_row['pitch']:
                        errors.append({
                            'type': 'Turn while moving',
                            'agv': name,
                            'timestamp': row['timestamp'],
                            'details': f"Moving from ({prev_row['X']},{prev_row['Y']}) to ({row['X']},{row['Y']}) while orientation changes from {prev_row['pitch']} to {row['pitch']}"
                        })
                
                # Turn check check
                elif dp != 0:
                    if dp not in [90, 180, 270]:
                        errors.append({
                            'type': 'Illegal turn',
                            'agv': name,
                            'timestamp': row['timestamp'],
                            'details': f"Turn angle {dp} is not a multiple of 90 degrees"
                        })
                
                # Load/unload check
                if loaded_changed:
                    if (prev_row['X'] != row['X'] or 
                        prev_row['Y'] != row['Y'] or 
                        prev_row['pitch'] != row['pitch']):
                        errors.append({
                            'type': 'Movement during load/unload',
                            'agv': name,
                            'timestamp': row['timestamp'],
                            'details': "Position or orientation changed during load/unload"
                        })
            prev_row = row
    
    return (len(errors) == 0, errors)

def calculate_metrics(df: pd.DataFrame, task_df: pd.DataFrame, map_df: pd.DataFrame):
    """
    Calculate task completion metrics
    :param df: Trajectory data
    :param task_df: Task data
    :param map_df: Map data
    :return: Total completion time, completed within 300 seconds, urgent tasks completed on time
    """
    # Prepare task queues
    task_queues = defaultdict(deque)
    for _, task in task_df.iterrows():
        queue_key = task['start_point']  # Group by starting point
        task_queues[queue_key].append(task.to_dict())
    
    # Prepare position information
    pickup_points = {}
    for _, row in map_df[map_df['type'] == 'start_point'].iterrows():
        if row['x'] == 1:  # Left side induction station - pickup point one cell to the right
            pickup_points[row['name']] = (row['x'] + 1, row['y'])
        else:  # Right side induction station - pickup point one cell to the left
            pickup_points[row['name']] = (row['x'] - 1, row['y'])
    
    drop_points = {}
    for _, row in map_df[map_df['type'] == 'end_point'].iterrows():
        x, y = row['x'], row['y']
        drop_points[row['name']] = [
            (x, y-1), (x, y+1), (x-1, y), (x+1, y)  # Four unload points
        ]
    
    # Track task status
    active_tasks = {}  # AGV current task: {agv: task_data}
    wrong_pickup_tasks = {}  # Wrong pickup task records: {agv: task_data}
    completed_tasks = []  # Completed tasks
    false_pickups = []  # False pickup records
    # Process trajectory in time order
    df = df.sort_values('timestamp')
    
    for _, row in df.iterrows():
        agv_name = row['name']
        
        # Load/unload processing
        if active_tasks.get(agv_name) and not row['loaded']:
            # Unload event
            task = active_tasks.pop(agv_name)
            end_point = task['end_point']
            drop_time = row['timestamp']

            # Check unload position
            valid_drop = (row['X'], row['Y']) in drop_points[end_point]
            on_time = True
            
            if task['priority'] == 'Urgent':
                remaining_time = task['remaining_time']
                on_time = (drop_time <= int(remaining_time))
            
            completed_tasks.append({
                'task_data': task,
                'drop_time': drop_time,
                'valid_drop': valid_drop,
                'on_time': on_time
            })

            if len(completed_tasks) == len(task_df):
                break
        
        if row['loaded'] and not active_tasks.get(agv_name) and not wrong_pickup_tasks.get(agv_name):
            # Pickup event
            pickup_point = None
            for point, (x, y) in pickup_points.items():
                if row['X'] == x and row['Y'] == y:
                    pickup_point = point
                    break
            
            if pickup_point and task_queues[pickup_point]:
                task = task_queues[pickup_point].popleft()
                if task['task_id'] != row['task_id']:
                    false_pickups.append({
                        'agv': agv_name,
                        'timestamp': row['timestamp'],
                        'details': f"Expected pickup task ID: {task['task_id']}, actual task ID: {row['task_id']}"
                    })
                    wrong_pickup_tasks[agv_name] = row['task_id']
                else:
                    active_tasks[agv_name] = task
    
    # Calculate metrics
    valid_completed = [t for t in completed_tasks if t['valid_drop']]
    max_time = max((t['drop_time'] for t in valid_completed), default=0)
    within_300 = sum(1 for t in valid_completed if t['drop_time'] <= 300)
    
    urgent_completed = [t for t in valid_completed 
                       if t['task_data']['priority'] == 'Urgent']
    urgent_on_time = sum(1 for t in urgent_completed if t['on_time'])
    
    return max_time, within_300, urgent_on_time, len(valid_completed), false_pickups
def check_collisions(df: pd.DataFrame):
    """
    Detect collision events and record detailed information
    :param df: Trajectory data
    :return: (collision count, collision detail list)
    """
    collisions = []  # List to store all collision events
    collisions_count = 0  # Total collision count
    timestamps = sorted(df['timestamp'].unique())
    
    # Static collision detection (same position at same time)
    for ts in timestamps:
        frame = df[df['timestamp'] == ts]
        
        # Group by position, find all positions with multiple AGVs
        grouped = frame.groupby(['X', 'Y'])
        for pos, group in grouped:
            if len(group) > 1:  # Multiple AGVs at same position
                agv_list = group['name'].tolist()
                # collisions_count += len(agv_list)  # Count once per AGV
                collisions_count += 1
                
                collisions.append({
                    'timestamp': ts,
                    'X': pos[0],
                    'Y': pos[1],
                    'type': 'vertex',
                    'AGVs': ", ".join(agv_list)
                })
    
    # Swapping collision detection (position exchange)
    for i in range(len(timestamps)-1):
        ts1, ts2 = timestamps[i], timestamps[i+1]
        df1 = df[df['timestamp'] == ts1][['name', 'X', 'Y']]
        df2 = df[df['timestamp'] == ts2][['name', 'X', 'Y']]
        
        # Create position mapping
        pos1 = {tuple(row[['X','Y']]): row['name'] for _, row in df1.iterrows()}
        pos2 = {tuple(row[['X','Y']]): row['name'] for _, row in df2.iterrows()}

        # Check all position swaps
        for (x1, y1), name1 in pos1.items():
            for (x2, y2), name2 in pos1.items():
                if name1 == name2:  # Skip self
                    continue
                
                # Check if positions are swapped
                if (pos2.get((x1, y1)) == name2 and 
                    pos2.get((x2, y2)) == name1):
                    
                    # Check if adjacent (Manhattan distance is 1)
                    if abs(x1 - x2) + abs(y1 - y2) == 1:
                        collisions_count += 1  # Count once for both AGVs
                        
                        collisions.append({
                            "timestamp": ts1,
                            "X": x1, 
                            "Y": y1,
                            "type": "swapping",
                            "AGVs": f"{name1}, {name2}"
                        })
    
    return collisions_count, collisions

def evaluate(agv_file, task_file, map_file):
    """
    Main evaluation function
    :param agv_file: AGV trajectory file path
    :param task_file: Task file path
    :param map_file: Map file path
    :return: Evaluation result dictionary
    """
    # Load data
    agv_df = pd.read_csv(agv_file)
    task_df = pd.read_csv(task_file)
    map_df = pd.read_csv(map_file)
    
    # Convert data types
    agv_df['loaded'] = agv_df['loaded'].astype(str).str.lower() == 'true'
    agv_df['Emergency'] = agv_df['Emergency'].astype(str).str.lower() == 'true'
    
    # Execute checks
    trajectory_valid, errors = check_agv_trajectory(agv_df)
    total_time, tasks_300, urgent_ontime, valid_completed_num, false_pickups = calculate_metrics(agv_df, task_df, map_df)
    collision_count, collisions = check_collisions(agv_df)

    # Return results
    return {
        'trajectory_valid': trajectory_valid,
        'trajectory_errors': errors,
        'total_completion_time': total_time,
        'tasks_completed_within_300': tasks_300,
        'urgent_tasks_ontime': urgent_ontime,
        'valid_completed_num': valid_completed_num,
        'false_pickup_count': len(false_pickups),
        'false_pickups': false_pickups,
        'collision_count': collision_count,
        'collisions': collisions
    }