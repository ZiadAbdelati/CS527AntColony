import numpy as np



class Player:
    def __init__(self, id, x, y, team):
        self.id = id
        self.pos = np.array([x, y], dtype=float)
        self.team = team.lower()  # normalize to lowercase



class Graph:
    def __init__(self, players, goal_pos=(100, 50)):
        self.players = players
        self.goal_pos = np.array(goal_pos, dtype=float)



    def distance(self, a, b):
        return np.linalg.norm(a.pos - b.pos)

   

    def angle_to_goal(self, player):
        # returning in radians right now but will fix later if we decide to use it
        vec_player_goal = self.goal_pos - player.pos
        return np.arctan2(vec_player_goal[1], vec_player_goal[0])



    def get_attackers(self):
        return [p for p in self.players if p.team == "offense"]



    def get_defenders(self):
        return [p for p in self.players if p.team == "defense"]



    def compute_pressure(self, position, defenders):
        pressure = 0.0
        for d in defenders:
            dist = np.linalg.norm(position - d.pos)
            pressure += 1 / (dist + 1e-6)
        return 1 / (1 + pressure)


    def is_offside(self, passer, receiver):
        ball_pos = passer.pos
        defenders = self.get_defenders()
        if not defenders:
            return False  # No defenders means no offside check possible
        
        # Sort defenders by closeness to their own goal
        goal_x = self.goal_pos[0]
        avg_attacker_x = np.mean([p.pos[0] for p in self.get_attackers()])
        attacking_right = self.goal_pos[0] > avg_attacker_x



        if attacking_right:
            defenders_sorted = sorted(defenders, key=lambda d: d.pos[0], reverse=True)
        else:
            defenders_sorted = sorted(defenders, key=lambda d: d.pos[0])



        # Last defender's X position not including the goalie
        # if only one defender then use that (in cases where goalie comes out of net possibly)

        if len(defenders_sorted) >= 2:
            last_defender_x = defenders_sorted[1].pos[0]
        else:
            last_defender_x = defenders_sorted[0].pos[0]
        midfield_x = 50.0
        # Offsides = ahead of last defender, ahead of ball, and in opponents half
        if attacking_right:
            in_opponent_half = receiver.pos[0] > midfield_x
            ahead_of_defenders = receiver.pos[0] > last_defender_x
            ahead_of_ball = receiver.pos[0] > ball_pos[0]
        else:

            in_opponent_half = receiver.pos[0] < midfield_x
            ahead_of_defenders = receiver.pos[0] < last_defender_x
            ahead_of_ball = receiver.pos[0] < ball_pos[0]
        return in_opponent_half and ahead_of_defenders and ahead_of_ball

    def xg_estimate(self, player, goal_pos, defenders):
        # only based off distance right now, will amend later
        dist = np.linalg.norm(player.pos - goal_pos)
        # Logistic decay (closer = higher xG)
        k = 0.1
        xg = 1 / (1 + np.exp(k * (dist - 30)))
        pressure = self.compute_pressure(player.pos, defenders)

        return xg * pressure