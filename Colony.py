import numpy as np
import random
from SoccerGraph import Graph, Player
from Ant import Ant

# Constants
EVAPORATION_RATE = 0.1      
PHEROMONE_DEPOSIT = 5.0     # Reward deposited along a successful scoring path
INITIAL_PHEROMONE = 1.0
SHOOT_RANGE = 25.0          # max distance from goal to attempt a shot
DEFENDER_BLOCK_RADIUS = 8.0 # how close a defender has to be to block a shot
 
 
class Colony:
    def __init__(self, graph, num_ants=10, num_iterations=50, alpha=1.0, beta=2.0):
        self.graph = graph
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
 
        # Pheromone matrix indexed by player id
        # all edges start at INITIAL_PHEROMONE

        attackers = graph.get_attackers()
        self.all_ids = [p.id for p in graph.players]
        self.pheromone = {
            i: {j: INITIAL_PHEROMONE for j in self.all_ids}
            for i in self.all_ids
        }
 
        # Tracking across iterations
        self.best_path = None
        self.best_score = -1.0  # xG of best path found so far
        self.history = []  # best_score after each iteration
 
    # Scoring / shot logic
    def _can_shoot(self, player):
        dist = np.linalg.norm(player.pos - self.graph.goal_pos)
        return dist <= SHOOT_RANGE
 
    def _shot_blocked(self, player):
        """
        Check whether a defender is close enough to block the shot.
        Could be extended later to use angles / shot trajectory.
        """
        defenders = self.graph.get_defenders()
        for d in defenders:
            if np.linalg.norm(player.pos - d.pos) <= DEFENDER_BLOCK_RADIUS:
                return True
        return False
 
    def _path_xg(self, path):
        """
        Estimate xG for a path's final shot.
        Uses the xg_estimate function from SoccerGraph.py.
        Returns 0 if the last player cannot shoot or is blocked.
        """
        if not path:
            return 0.0
 
        shooter = path[-1]
 
        if not self._can_shoot(shooter):
            return 0.0

        return self.graph.xg_estimate(shooter, self.graph.goal_pos, self.graph.get_defenders())
 
    # Pheromone update
 
    def _evaporate(self):
        """
        Decay all pheromones each iteration - simulates defense adapting to stop
        previously successful passing lanes over time.
        """
        for i in self.all_ids:
            for j in self.all_ids:
                self.pheromone[i][j] = max(
                    0.01,  # floor so edges never become completely dead
                    self.pheromone[i][j] * (1 - EVAPORATION_RATE)
                )
 
    def _deposit(self, path, xg):
        """
        Reinforce pheromones along the edges of a successful scoring path.
        Higher xG = stronger deposit, rewarding paths that create better shots.
        """
        if xg <= 0 or len(path) < 2:
            return
 
        deposit_amount = PHEROMONE_DEPOSIT * xg  # scale reward by shot quality
 
        for k in range(len(path) - 1):
            a = path[k].id
            b = path[k + 1].id
            self.pheromone[a][b] += deposit_amount
            self.pheromone[b][a] += deposit_amount  # passes can go either way
 
    # Main simulation loop
 
    def run(self):
        """
        Each iteration: spawn ants, let them find paths, evaluate xG,
        evaporate pheromones, deposit rewards on good paths.
        Returns the best path and its xG found across all iterations.
        """
        attackers = self.graph.get_attackers()
        defenders = self.graph.get_defenders()
 
        if not attackers:
            print("No offensive players found - check team labels.")
            return None, 0.0
 
        for iteration in range(self.num_iterations):
            iteration_paths = []
            iteration_scores = []
 
            for _ in range(self.num_ants):
                # Each ant starts from a random attacker
                start = random.choice(attackers)
 
                ant = Ant(
                    graph=self.graph,
                    pheromone=self.pheromone,
                    alpha=self.alpha,
                    beta=self.beta
                )
 
                # Traverse until stuck or max steps reached
                path = ant.traverse(
                    start=start,
                    defenders=defenders
                )
                xg = self._path_xg(path)

                iteration_paths.append(path)
                iteration_scores.append(xg)
 
                # Track best overall path
                if xg > self.best_score:
                    self.best_score = xg
                    self.best_path = path
 
            self.history.append(self.best_score)

            # Evaporate first, then reward good paths
            self._evaporate()
 
            for path, xg in zip(iteration_paths, iteration_scores):
                self._deposit(path, xg)
 
        return self.best_path, self.best_score
 
    def print_best_path(self):
        if not self.best_path:
            print("No valid scoring path found.")
            return
 
        print("\n--- Best Passing Sequence Found ---")
        for i, player in enumerate(self.best_path):
            prefix = "START" if i == 0 else f"Pass {i}"
            print(f"  {prefix}: Player {player.id} at {tuple(player.pos.astype(int))}")
        print(f"  SHOOT (xG: {self.best_score:.3f})")
        print("-----------------------------------\n")