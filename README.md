# S-MAPD

This repository contains the implementation of our S-MAPD (Sequential Multi-Agent Pickup and Delivery) algorithm, along with baseline algorithms TP and CENTRAL.

## Algorithms

- **S-MAPD Algorithm** (`Planner.py`): Our proposed algorithm.
- **TP Algorithm** (`TP.py`): Token Passing baseline algorithm for S-MAPD.
- **CENTRAL Algorithm** (`CENTRAL.py`): Centralized planning baseline.

## Quick Start

### 1. Configuration

Edit `settings.json` to configure your experiment:

```json
{
  "scene_size": "small",
  "map_distribution": "sorting_center",
  "num_agvs": 10,
  "num_tasks": 500,
  "task_distribution": "uniform"
}
```

#### Configuration Rules:

**Scene Size:**
- `small`: Map size 35x21, agent options: 10, 20, 30, 40, 50, tasks: 500
- `large`: Map size 70x43, agent options: 100, 200, 300, tasks: 1000
  - Note: Only `sorting_center` map distribution is supported for large scenes
  - Only our S-MAPD algorithm (Planner.py) supports large scenes

**Map Distribution:**

- `sorting_center`: Sorting center layout
- `logistics_warehouse`: Logistics warehouse layout  
- `workshop`: Workshop layout

**Task Distribution:**
- Currently only `uniform` distribution is supported

### 2. Running Algorithms

**Our S-MAPD Algorithm:**
```bash
python Planner.py
```

**TP Algorithm:**
```bash
python TP.py
```

**CENTRAL Algorithm:**
```bash
python CENTRAL.py
```

## Data Files

The `data/` directory contains pre-generated map and task CSV files for all supported configurations. Files are named according to the pattern:
- Maps: `map_{scene_size}_{map_distribution}_AGV{num_agvs}.csv`
- Tasks: `task_{scene_size}_{map_distribution}_{task_distribution}_{num_tasks}.csv`

## Output

Each algorithm generates:
1. **Trajectory CSV**: AGV movement logs in `data/agv_trajectory_*.csv`
2. **Console Output**: Performance metrics including:
   - Path validity
   - Collision count
   - Makespan
   - Task completion rate
   - Service time
   - Runtime in total

## Trajectory Evaluation

The built-in evaluation module (`evaluate.py`) automatically validates:
- Trajectory legality (movement rules, turns, load/unload constraints)
- Task completion correctness
- Collision detection (vertex and edge collisions)
- Performance metrics

## File Structure

- `Planner.py` - Main implementation of our S-MAPD algorithm
- `TP.py` - Token Passing baseline implementation
- `CENTRAL.py` - Centralized planning baseline
- `Agent.py` - AGV agent class
- `Task.py` - Task definition and management
- `Allocator.py` - Task allocation strategies (BSA)
- `CBS.py` - Conflict-Based Search implementations with variants
- `PickupStation.py` - Pickup station management
- `PathUtils.py` - Path planning utilities
- `DataLoader.py` - Data loading from CSV files
- `config_manager.py` - Configuration management
- `evaluate.py` - Performance evaluation
- `settings.json` - Configuration file
- `data/` - Pre-generated map and task datasets
- `cache/` - Precomputed path cost maps for performance optimization (contains JSON files with precalculated distances from all map positions to all endpoints)
