import numpy as np
import random
from SoccerGraph import Graph

class Ant:
    def __init__(self, graph, pheromone, alpha, beta):
        self.graph = graph
        self.pheromone = pheromone # stores pheromones / how desirable of a pass
        self.alpha = alpha # pheromone importance
        self.beta = beta # heuristic importance 
        self.path = []

    def _line_point_dist(self, a, b, p):
        # distance from defender p to line segment a->b
        ap = p - a # passer to defender
        ab = b - a # passer to receiver

        # defender position relative to pass 
        t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-6)
        # t < 0 = behind passer, t > 1 beyond receiver; no risk
        t = max(0, min(1, t))
        closest = a + t * ab
        return np.linalg.norm(p - closest)

    def choose_next(self, current, candidates, defenders):
        probs = []
        total = 0.0
        max_gain = 20.0

        for c in candidates:
            
            # look at how pheromones and heuristic affect pass choices
            tau = self.pheromone[current.id][c.id]

            # distance (short passes preferred)
            pass_dist = np.linalg.norm(current.pos - c.pos)
            h_pass = 1 / (1 + pass_dist)

            # passes towards goal, set max_gain to normalize the scores
            # we normalize so "hail mary" passes do not dominate
            # output in (0,1)
            # can also apply slight penalty to passes longer than max if wanted 
            curr_dist = np.linalg.norm(current.pos - self.graph.goal_pos)
            potential_dist = np.linalg.norm(c.pos - self.graph.goal_pos)
            gain = curr_dist - potential_dist
            h_goal = 1 / (1 + np.exp(-gain / max_gain))

            # avoid defenders close to receiver and passing lane
            pressure = self.graph.compute_pressure(c.pos, defenders)

            # can weight eta differently
            eta = (
                    0.35 * h_goal +
                    0.30 * h_pass +
                    0.35 * pressure
            )

            eta = max(eta, 1e-6) # preventing collapse

            # append probs using ACO rule
            val = (tau ** self.alpha) * (eta ** self.beta)
            probs.append((c, val))
            total += val

        # dummy check if EVERYONE else is offsides, then keep ball
        if total == 0:
            valid = [c for c, _ in probs]
            return random.choice(valid) if valid else None
        
        # roulette wheel selection
        r = random.uniform(0, total)
        cum = 0

        # larger val = larger prob of being chosen
        for c, val in probs:
            cum += val
            if cum >= r:
                return c

        return probs[-1][0]

    def traverse(self, start, defenders, max_steps=10):
        current = start
        self.path = [current]
        
        for _ in range(max_steps):
            # Calculate the xG of the player currently holding the ball
            current_xg = self.graph.xg_estimate(current, self.graph.goal_pos, defenders)

            if current_xg > 0.4:  # Only consider shooting if xG is non-negligible
                if random.random() < current_xg:
                    break # Stop passing and shoot
            

            # Get legal candidates (not offside, not already in path)
            candidates = [p for p in self.graph.get_attackers() 
                         if p not in self.path and not self.graph.is_offside(current, p)]

            next_node = self.choose_next(current, candidates, defenders)

            if next_node is None:
                break
            
            self.path.append(next_node)
            current = next_node

        # Return the final path
        return self.path