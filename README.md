This project models offensive decision-making in soccer using Ant Colony Optimization (ACO). Attackers are treated as nodes in a graph, and simulated "ants" explore passing sequences to find high-quality scoring opportunities. 

The system learns which passing paths are most effective based on expected goals (xG), while accounting for defensive pressure and offside constraints. 

**Key Idea**
We model a soccer possession as a search problem: 
- Each ant simulates a sequence of passes between attacking players
- Decisions are guided by:
  - distance between players
  - progress toward goal
  - defensive pressure
  - pheromone trails (learned from past success)
- The final shot is evaluated using an xG-based reward function

Over time, the system learns which passing sequences are most likely to lead to high-quality shots.

**Project Structure**
.
├── SoccerGraph.py   # player representation + geometry + xG model
├── Ant.py           # ant traversal + decision-making (passing + shooting)
├── Colony.py        # ACO loop, pheromone updates, evaluation
├── Test.py          # run experiments

**How It Works**
1. Initialization
   - Players are placed on a 2D field
   - Pheromone levels are initialized between all players
2. Simulation Loop
   - Ants generate passing sequences
   - Each sequence ends in shot attempt
   - The shot is scored using xG
   - Pheromones updated based on success
3. Learning
   - Better sequences -> more pheromone
   - Poor sequences -> fade over time (evaporate)

**Core Components**
- Ant Colony Optimization (ACO): learns effective passing paths
- Spatial Heuristics: incorporates distance, goal progression, and pressure
- Offside Rule: enforces realistic passing constraints
- xG Model: estimates shot quality based on distance and defensive pressure
- Stochastic Shooting: players shoot probabilistically based on xG

**Possible Extensions**
- Add shot angle
- Model defender movement
- Learn heuristic weights instead of hand-tuning

**How to Run**
python main.py

