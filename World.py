from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, List, Tuple
import numpy as np
import math
from LQR import LQR
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from POMDP.pomcp import POMCP
from enum import Enum, auto
import random



#3.7 meter breite 

# ────────────────────────────────────────────────────────────────────────────────
# 1.  PARAM – static configuration
# ────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Param:
    

    # bounds (y‑axis is lateral, x‑axis longitudinal)
    pos_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0,  7.4),   # y  
                                                                (0.0, 100.0))   # x  [m]
    vel_bounds: Tuple[float, float] = (0.0, 8.0)   # [m/s]
    acc_bounds: Tuple[float, float] = (-2.0, 2.0)   # [m/s²]
    heading_bounds: Tuple[float, float] = (-math.pi/6, math.pi/6) #rad

    # start / goal (long., lat.)
    s_start: Tuple[float, float] = (0.0, 1.85)   # x, y
    s_goal:  Tuple[float, float] = (100.0, 1.85)#(100.0,  5.55)


    lane_centres: Tuple[float, float] = (1.85, 5.6)  # two‑lane centres (approx ±0.5)


# ────────────────────────────────────────────────────────────────────────────────
# 2.  MDP
# ────────────────────────────────────────────────────────────────────────────────
@dataclass
class HumanMDP:
    """Discretised state and action grids for value‑iteration look‑up."""
    param: Param

    STATES: np.ndarray = field(init=False)
    ACTIONS: np.ndarray = field(init=False)
    POS: np.ndarray = field(init=False)
    VEL: np.ndarray = field(init=False)

    def __post_init__(self):
        data = np.load('world.npz')
        self.STATES = data['states']
        self.ACTIONS = data['actions']
        self.Q = data['Q']



class State(Enum):
    LANE_1 = auto()
    LANE_1_PLUS = auto()
    LANE_2 = auto()
    LANE_2_MINUS = auto()

class Action(Enum):
    INDICATE = auto()
    CHANGE_LANE = auto()

@dataclass
class EgoMDP:
    STATES: np.ndarray = field(init=False)
    ACTIONS: np.ndarray = field(init=False)
    busy: bool = field(init=False, default=False)
    lane_y: dict = field(init=False)
    tol: float = field(init=False, default=0.2)

    def __init__(self, init_state=State.LANE_1):
        # MDP components
        self.states = list(State)
        self.actions = list(Action)
        self.current_state = init_state
        self.desiredState = init_state
        self.indicating = False

        self.busy = False
        param = Param()
        self.lane_y = {
            State.LANE_1: param.lane_centres[0],
            State.LANE_1_PLUS: param.lane_centres[0] + 0.6,
            State.LANE_2: param.lane_centres[1],
            State.LANE_2_MINUS:param.lane_centres[1] - 0.6
        }

        # Transition Probabilities P(s'|s,a)
        self.P = {
            (State.LANE_1, Action.INDICATE):      {State.LANE_1_PLUS: 1.0},
            (State.LANE_1_PLUS, Action.CHANGE_LANE): {State.LANE_2: 1.0},
            (State.LANE_2, Action.INDICATE):      {State.LANE_2_MINUS: 1.0},
            (State.LANE_2_MINUS, Action.CHANGE_LANE): {State.LANE_1: 1.0},
        }

    def transition(self, state: State, action: Action) -> dict:
            return self.P.get((state, action), {state: 1.0})
        
    def update_busy(self, ego_y: float):

        if self.busy:
            target_y = self.lane_y[self.desiredState]
            if abs(ego_y - target_y) <= self.tol:
                self.busy = False
                self.current_state = self.desiredState


        #
        if self.current_state in (State.LANE_1_PLUS, State.LANE_2_MINUS):
            self.time_in_LC += 0.5
            self.indicating = True
        else:
            self.time_in_LC = 0.0
            self.indicating = False
        return self.busy
    





# ────────────────────────────────────────────────────────────────────────────────
# 3.  Vehicle definitions
# ────────────────────────────────────────────────────────────────────────────────
class Vehicle:
    """Base vehicle (kinematic)."""
    L: float = 2.7           # wheel‑base


    delta_max: float = np.deg2rad(30)

    def __init__(self, x: float, y: float, theta: float, v: float):
        self.x, self.y, self.theta, self.v = x, y, theta, v
        self.a_max, self.b_max = 2.0, 4.0
        self.vx = self.v * math.cos(self.theta)
        self.vy = self.v * math.sin(self.theta)

    # ‑‑‑  (simplified bicycle) ‑‑‑
    def update(self, delta: float, a: float, dt: float = 0.5):
        delta = np.clip(delta, -self.delta_max, self.delta_max)
        self.theta += (self.v / self.L) * np.tan(delta) * dt
        self.x     += self.v * np.cos(self.theta) * dt
        self.y     += self.v * np.sin(self.theta) * dt
        self.v      = np.clip(self.v + a * dt, 0.0, 14.0)
        self.vx = self.v * math.cos(self.theta)
        self.vy = self.v * math.sin(self.theta)

    # ‑‑‑ Corners for collision check ‑‑‑
    def get_corners(self, CAR_LENGTH = 4.5, CAR_WIDTH = 1.8):
          # Meter
           # Meter
        rel  = np.array([[-CAR_LENGTH/2,  CAR_WIDTH/2], [ CAR_LENGTH/2,  CAR_WIDTH/2],
                         [ CAR_LENGTH/2, -CAR_WIDTH/2], [-CAR_LENGTH/2, -CAR_WIDTH/2]])
        rot  = np.array([[np.cos(self.theta), -np.sin(self.theta)],
                         [np.sin(self.theta),  np.cos(self.theta)]])
        return rel @ rot.T + np.array([self.x, self.y])


class EgoCar(Vehicle):
    def __init__(self, param: Param, velocity):
        super().__init__(x=param.s_start[0],
                         y=param.s_start[1],
                         theta=0.0,
                         v=velocity)
        self.target_lane = 1            # index into param.lane_centres
        self.busy        = True         # lane‑change running flag
        self.indicating  = False
        self.goal = np.array([ 
                      velocity,
                      param.s_start[1], 0], dtype=float)
        self.controller = LQR()
        self.controller.updateGoal(self.goal)
        self.action = 0
        self.egoMDP = EgoMDP()

    @property
    def y_des(self):
        return Param().lane_centres[self.target_lane]
    
    def control(self, newGoal, dt = 0.5):
        z = np.array([self.vx ,self.y, self.vy])


        self.controller.updateGoal(newGoal)
        z_next, u = self.controller.updateState(z)

        vx_new, y_new, vy_new = z_next
        #limit tilt
        max_tilt = math.radians(30)
        vy_max = abs(vx_new * math.tan(max_tilt))
        vy_new = np.clip(vy_new, -vy_max, vy_max)

        self.vx, self.y, self.vy = vx_new, y_new, vy_new 
        self.x += self.vx *dt
        self.v = math.hypot(self.vx, self.vy)
        self.theta = math.atan2(self.vy, self.vx)
        

       


BEHAVIOURS = ["aggressive", "calm"]
class HumanCar(Vehicle):
    
    DEFAULT_IDM_PARAMS = {
        "calm": {
            "T": 1.0,    # Reaktionszeit
            "s0": 3,   # Mindestabstand
            "a_max": 2,  # Maximale Beschleunigung
            "b": 2.0,    # Komfortables Bremsen
            "v0": 3 ,  # Zielgeschwindigkeit (m/s)
            "delta": 4,
            "n0": 1.5,
            "c": 1.0
        },
        "aggressive": {
            "T": 1.0,    # Reaktionszeit
            "s0": 3,   # Mindestabstand
            "a_max": 2,  # Maximale Beschleunigung
            "b": 2.0,    # Komfortables Bremsen
            "v0": 3,  # Zielgeschwindigkeit (m/s)
            "delta": 4,
            "n0": 1.5,
            "c": 0.0
        }
    }
    # DEFAULT_IDM_PARAMS = {
    #     "calm": {
    #         "T": 1.0,    # Reaktionszeit
    #         "s0": 10,   # Mindestabstand
    #         "a_max": 2,  # Maximale Beschleunigung
    #         "b": 2.0,    # Komfortables Bremsen
    #         "v0": 8 ,  # Zielgeschwindigkeit (m/s)
    #         "delta": 4,
    #         "n0": 1.5,
    #         "c": 0.9
    #     },
    #     "aggressive": {
    #         "T": 1.0,    # Reaktionszeit
    #         "s0": 10,   # Mindestabstand
    #         "a_max": 2,  # Maximale Beschleunigung
    #         "b": 2.0,    # Komfortables Bremsen
    #         "v0": 8,  # Zielgeschwindigkeit (m/s)
    #         "delta": 4,
    #         "n0": 1.5,
    #         "c": 0.1
    #     }
    # }
    def __init__(self, x, y, theta, v, id, behavior = "calm"):
        super().__init__(x=x,
                         y=y,
                         theta=0.0,
                         v=v)
        self.behavior = behavior
        self.idm_params = self.DEFAULT_IDM_PARAMS.get(behavior, self.DEFAULT_IDM_PARAMS["calm"])
        self.id = id
        self.belief = None

    def control(self, egoCar, humans):
        acc_disc = np.linspace(-2, 2, 5)
        T = self.idm_params["T"]
        s0 = self.idm_params["s0"]
        a_max = self.idm_params["a_max"]
        b = self.idm_params["b"]
        v0 = self.idm_params["v0"]
        delta = self.idm_params["delta"]
        n0 = self.idm_params["n0"]
        c = self.idm_params["c"]

       
        leaders = [egoCar] + [h for h in humans if h is not self]
        #egoCorners = egoCar.get_corners()
        selfCorners = self.get_corners()
        # Find nearest vehicle in front on same lane
        best_gap = float('inf')
        best_dv  = 0.0
        for lead in leaders:
            leadCorners = lead.get_corners()
            dx, dy, inFront = getDistance(leadCorners,selfCorners)
            # same lane if dy small and in front
            if inFront and dy == 0:
                if dx < best_gap:
                    best_gap = dx
                    best_dv  = egoCar.v - self.v
            elif lead == egoCar and dy <= n0 and dy > 0 and inFront and dx < 20:
                if dx < best_gap:
                    if np.random.binomial(n=1,p=c) :
                        best_gap = dx
                        best_dv = egoCar.v - self.v

        # No leader
        if best_gap == float('inf'):
            gap     = 1e6
            delta_v = 0.0
        else:
            gap     = best_gap
            delta_v = best_dv




        s_star = s0 + self.v * T + (self.v * delta_v) / (2 * math.sqrt(a_max * b))
        acc = a_max * (1 - (self.v / v0) ** delta - (s_star / gap) ** 2)
        acc =  max(-2*b, min(a_max, acc))

        idx = np.abs(acc_disc - acc).argmin()
        return acc_disc[idx]
        #self.update(delta=0.0, a= acc)



    def yield_corners(self):
        corners = self.get_corners()
        x_max = corners[:,0].max()   # x max
        y_max = corners[:,1].max()   # y max
        y_min = corners[:,1].min()   # y min

        forceyield = np.array[(x_max, y_max), (x_max + self.idm_params["s0"], y_max),
                        (x_max, y_min), (x_max + self.idm_params["s0"], y_min)]
        
        selectiveyield_left = np.array[(x_max, y_max), (x_max + self.idm_params["s0"], y_max),
                                       (x_max, y_max + self.idm_params["n0"]), (x_max + self.idm_params["s0"], y_max+ self.idm_params["n0"]) ]

        selectiveyield_right = np.array[(x_max, y_min), (x_max + self.idm_params["s0"], y_min),
                                       (x_max, y_min - self.idm_params["n0"]), (x_max + self.idm_params["s0"], y_min- self.idm_params["n0"]) ]


        return forceyield, selectiveyield_left, selectiveyield_right
    


# ────────────────────────────────────────────────────────────────────────────────
# 4.  World class 
# ────────────────────────────────────────────────────────────────────────────────
class World:
   

    def __init__(self, param: Param = None):
        self.param = param or Param()
        self.mdp   = HumanMDP(self.param)
       # self.egoMDP = EgoMDP()
        self.ego   = EgoCar(self.param, velocity = 1)
        #self.ego   = EgoCar(self.param, velocity = 8)
        self.humans: List[HumanCar] = []

        
        self.one_step_model: Callable = self._one_step
        self.dt = 0.5
        self.time = 0
        self.next_human_id = 0
        self.planner = None
        self.oppoID = None
        

    # ‑‑‑ public API to add a new human agent ‑‑‑
    def add_human(self, x: float, y: float, v: float, behavior: str = None):
        if behavior is None:
            behavior = random.choice(BEHAVIOURS)

        self.humans.append(HumanCar(x=x, y=y, theta=0.0, v=v, id=self.next_human_id, behavior= behavior))
        self.next_human_id += 1

    # ‑‑‑ unified dynamics for ego+humans
    def _one_step(self, ego_action: int, human_actions: List[int], egoCar: EgoCar= None, humans: List[HumanCar]= [], particles: List[Tuple[int, float]] = []):
        # 1.  Ego primitive handling (simplified)
        if egoCar == None and humans == []:
            egoCar = self.ego
            humans = self.humans

        goal = egoCar.goal
        egoCar.egoMDP.update_busy(egoCar.y)
    
        if ego_action == -1:   # accelerate
            goal[0] -= 1 
        elif ego_action == 0: # brake
            goal[0] += 0
        elif ego_action == 1: # keep velocity
            goal[0] +=  1
        elif ego_action == 2 and egoCar.egoMDP.busy == False:  #Indicate
            dictState = egoCar.egoMDP.transition(egoCar.egoMDP.current_state, Action.INDICATE)
            nextState = list(dictState.keys())
            goal[1] = egoCar.egoMDP.lane_y[nextState[0]]
            egoCar.egoMDP.desiredState = nextState[0]
            egoCar.egoMDP.busy = True
            
            egoCar.indicating = True
        elif ego_action == 3 and egoCar.egoMDP.busy == False:  #Change Lane
            dictState = egoCar.egoMDP.transition(egoCar.egoMDP.current_state, Action.CHANGE_LANE)
            nextState = list(dictState.keys())
            goal[1] = egoCar.egoMDP.lane_y[nextState[0]]
            egoCar.egoMDP.busy = True   
            egoCar.egoMDP.desiredState = nextState[0]
            egoCar.indicating = False

  
        egoCar.goal = goal
        egoCar.control(goal)

        
        
        for i, h in enumerate(humans):
            h.update(0, human_actions[i])
            beliefs = getBelief(particles)
            #Für jeden Human belief berechnen



        reward = self.reward(egoCar, humans, beliefs)
        egoCar.action = ego_action
        return egoCar, humans, reward
    

    def legal_actions(self, egoCar: EgoCar= None, humans: List[HumanCar]= []) -> List[int]:
        """Return list of action‑IDs that are *currently* allowed."""
        if egoCar == None and humans == []:
            egoCar = self.ego
            humans = self.humans
        acts: List[int] = []
        v = egoCar.controller.z_goal[0]

        # Longitudinal
        if v < self.param.vel_bounds[1] - 1e-3:   # accelerate only if below vmax
            acts.append(1)             # ID 1
        if v > 1e-3:                    # brake only if moving
            acts.append(-1)              # ID -1
        acts.append(0)                  # keep velocity always legal (ID 0)


        if not egoCar.egoMDP.busy:
            for action in [Action.INDICATE, Action.CHANGE_LANE]:
                if (egoCar.egoMDP.current_state, action) in egoCar.egoMDP.P:
                    if action == Action.CHANGE_LANE:
                        acts.append(3) 
                    if action == Action.INDICATE:
                        acts.append(2)   


        return acts

    # simple reward example
    def reward(self, egoCar: EgoCar= None, humans: List[HumanCar]= [], beliefs: List[Tuple[float,float,float]] = None) -> float:

        if egoCar == None and humans == []:
            egoCar = self.ego
            humans = self.humans

        reward = 0

        if self.collision(egoCar, humans):
            reward -= 10000
        
        if self.collision(egoCar, humans, test= True):
            reward -= 100

        
        self.time += self.dt
        #reward -= 0.1 * self.dt

        lane_error = abs(egoCar.y - egoCar.y_des)
        reward -= lane_error * 10

        reward += egoCar.x *0.1

        v_max = Param().vel_bounds[1]
        reward -= 5.0 * abs(v_max - egoCar.v)
        reward += 1.0 * egoCar.v


        #reward +=  egoCar.v  
        #t = egoCar.egoMDP.time_in_LC
        #reward -= t * 50

        if egoCar.egoMDP.indicating:
            for human in humans:
                if human.id  in self.oppoID:
                    p0, p1, p2 = beliefs[human.id] #get the belief of the opponent
                    v = egoCar.v
                    # Reward‐Term: p1 * (+v)  –  (p0+p2) * (+v)
                    reward += 20 * p1 * v
                    reward -= 20 * (p0 + p2) * v

        

        return reward

    #
    def collision(self,  egoCar: EgoCar= None, humans: List[HumanCar]= [], test = False) -> bool:
        # 1.  Ego primitive handling (simplified)
        if egoCar == None and humans == []:
            egoCar = self.ego
            humans = self.humans
        
        if test == True:
            CAR_LENGTH = 10
            CAR_WIDTH = 2.8
        else:
            CAR_LENGTH = 5
            CAR_WIDTH = 2

        egoCorner = egoCar.get_corners(CAR_LENGTH, CAR_WIDTH )
        for human in humans:
            humanCorner = human.get_corners(CAR_LENGTH, CAR_WIDTH )

            dx, dy, _ = getDistance(egoCorner, humanCorner)
            if dx == 0 and dy == 0:
                return True

        return False
    
    # selects the nearest human agents and returns  a List of max. 2 oppoIDs
    def selectOpponent(self):

        egoCar = self.ego
        egoCorner = egoCar.get_corners()
        y_ego = egoCar.y_des #get the desired y_pos of the ego
        tol = 0.2

        distances = []

        for human in self.humans:
            if abs(human.y - y_ego) <= tol:
                humanCorner = human.get_corners()
                dx, dy, egoInFront = getDistance(egoCorner, humanCorner)
                if egoInFront and dx < 20:
                    dist = math.hypot(dx, dy)
                    distances.append((dist, human.id))

        if not distances:
            return []
        
        distance, opponent = min(distances, key=lambda t: t[0])
        return [opponent]
        



def getDistance(egoCorner, humanCorner):

    #Min/Max 
    xEgo_min, xEgo_max = egoCorner[:,0].min(), egoCorner[:,0].max()
    yEgo_min, yEgo_max = egoCorner[:,1].min(), egoCorner[:,1].max()
    xH_min, xH_max   = humanCorner[:,0].min(), humanCorner[:,0].max()
    yH_min, yH_max   = humanCorner[:,1].min(), humanCorner[:,1].max()

    # distance in x
    if xEgo_max < xH_min:
        dx = xH_min - xEgo_max
    elif xH_max < xEgo_min:
        dx = xEgo_min - xH_max
    else:
        dx = 0.0

    # distance in y
    if yEgo_max < yH_min:
        dy = yH_min - yEgo_max
    elif yH_max < yEgo_min:
        dy = yEgo_min - yH_max
    else:
        dy = 0.0

    # ego in front?
    x_center_ego   = (xEgo_min + xEgo_max) / 2.0
    x_center_human = (xH_min   + xH_max)   / 2.0
    ego_in_front = x_center_ego > x_center_human

    return round(dx,2), round(dy,2), ego_in_front
    

def getBelief(particles):

    beliefs = []
    for agent_idx, plist in enumerate(particles):
        total = len(plist)
        if total == 0:
           # No particles
            beliefs.append((0.0, 0.0, 0.0))
            continue

        # count k = 0, 1 and 2
        counts = Counter(p[0] for p in plist)
        p0 = counts.get(0, 0) / total
        p1 = counts.get(1, 0) / total
        p2 = counts.get(2, 0) / total

        beliefs.append((p0, p1, p2))

    return beliefs



def print_belief_percentages(particles):

    for agent_idx, particles in enumerate(particles):
        total = len(particles)
        if total == 0:
            print(f"Agent {agent_idx}: no particles")
            continue

        # 1) Verteilung über k
        k_vals = [p[0] for p in particles]
        counts = Counter(k_vals)

        # 2) Verteilung in Prozent ausgeben
        print(f"Agent {agent_idx}:")
        for k, cnt in counts.most_common():
            pct = cnt / total * 100
            print(f"  Level k={k}: {pct:.2f}%")

        # 3) Statistik über lambda
        lam_vals = [p[1] for p in particles]
        mean_lam = np.mean(lam_vals)
        std_lam  = np.std(lam_vals)
        print(f"  lambda-avg: {mean_lam:.3f}, std: {std_lam:.3f}\n")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

def plot_cars(ego_car, humans, ax, t):
    ax.clear()

    # --- Achsen-Limits berechnen ---
    x_min, x_max = ego_car.x - 30, ego_car.x + 30
    y_min, y_max = 0.0, 7.4
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')

    # --- Straße zeichnen ---
    lane_width = 3.7
    # Mittelpunkte der Fahrspuren
    y1_center = 1.85
    y2_center = 5.55
    # Ränder der Fahrspuren
    y1_bot = y1_center - lane_width/2   # = 0
    y1_top = y1_center + lane_width/2   # = 3.7
    y2_bot = y2_center - lane_width/2   # = 3.7
    y2_top = y2_center + lane_width/2   # = 7.4

    # 1) Graue Flächen
    ax.add_patch(Rectangle((x_min, y1_bot),
                           width=x_max-x_min, height=lane_width,
                           facecolor='lightgray', edgecolor='none', zorder=0))
    ax.add_patch(Rectangle((x_min, y2_bot),
                           width=x_max-x_min, height=lane_width,
                           facecolor='lightgray', edgecolor='none', zorder=0))

    # 2) Weiße Randlinien (durchgehend)
    ax.plot([x_min, x_max], [y1_bot, y1_bot],
            color='white', linewidth=2, zorder=1)
    ax.plot([x_min, x_max], [y2_top, y2_top],
            color='white', linewidth=2, zorder=1)

    # 3) Mittellinie (gestrichelt)
    ax.plot([x_min, x_max], [y1_top, y1_top],
            color='white', linewidth=2, linestyle=(0, (10, 10)), zorder=1)

    # --- Fahrzeuge zeichnen ---
    # Ego
    ego_corners = ego_car.get_corners(CAR_LENGTH=4.5, CAR_WIDTH=1.8)
    ax.add_patch(Polygon(ego_corners, closed=True,
                         facecolor='tab:blue', edgecolor='black', alpha=0.5, zorder=2))
    # Optional „Sensorbereich“ um Ego
    ego_outline = ego_car.get_corners(CAR_LENGTH=8, CAR_WIDTH=2.4)
    ax.add_patch(Polygon(ego_outline, closed=False,
                         edgecolor='tab:blue', alpha=0.5, zorder=2))
            # ID-Label im Zentrum des Vierecks
    centroid = np.mean(ego_corners, axis=0)
    ax.text(centroid[0], centroid[1],
            "Ego",
            color='black',
            fontsize=9,
            fontweight='bold',
            ha='center', va='center',
            zorder=3)

    # Humans
    for i, human in enumerate(humans):
        h_corn = human.get_corners(CAR_LENGTH=4.5, CAR_WIDTH=1.8)
        ax.add_patch(Polygon(h_corn, closed=True,
                             facecolor='tab:red', edgecolor='black', alpha=0.5, zorder=2))
        h_outline = human.get_corners(CAR_LENGTH=8, CAR_WIDTH=2.4)
        ax.add_patch(Polygon(h_outline, closed=False, facecolor='tab:red',
                             edgecolor='tab:red', alpha=0.5, zorder=2))
        # ID-Label im Zentrum des Vierecks
        centroid = np.mean(h_corn, axis=0)
        ax.text(centroid[0], centroid[1],
                str(human.id),
                color='black',
                fontsize=9,
                fontweight='bold',
                ha='center', va='center',
                zorder=3)
        

    # --- Info-Panel und Beschriftungen ---
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    #ax.set_title(f"Simulation time: {t:4.1f} s")
    ax.grid(False)  # optional: das Raster über der Straße stört hier

    # Ego-Status
    ax.text(0.02, 2.98, 
            (f"EGO\n"
             f"x={ego_car.x:5.1f} m\n"
             f"y={ego_car.y:5.1f} m\n"
             f"v={ego_car.v:5.1f} m/s\n"
             f"θ={np.degrees(ego_car.theta):5.1f}°"),
            transform=ax.transAxes, fontsize=9, color='tab:blue',
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='tab:blue', lw=0.8))

    # # Human-Status (erste Human-Car)
    # if humans:
    #     human = humans[0]
    #     ax.text(0.98, 2.98,
    #             (f"HUMAN\n"
    #              f"x={human.x:5.1f} m\n"
    #              f"y={human.y:5.1f} m\n"
    #              f"v={human.v:5.1f} m/s\n"
    #              f"θ={np.degrees(human.theta):5.1f}°"),
    #             transform=ax.transAxes, fontsize=9, color='tab:red',
    #             va='top', ha='right',
    #             bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='tab:red', lw=0.8))

    plt.pause(0.01)

def plot_cars_history(ego_car, humans, ax, t):


    # --- Achsen-Limits berechnen ---
    x_min, x_max = 0, 120
    y_min, y_max = 0.0, 7.4
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')

    # --- Straße zeichnen ---
    lane_width = 3.7
    # Mittelpunkte der Fahrspuren
    y1_center = 1.85
    y2_center = 5.55
    # Ränder der Fahrspuren
    y1_bot = y1_center - lane_width/2   # = 0
    y1_top = y1_center + lane_width/2   # = 3.7
    y2_bot = y2_center - lane_width/2   # = 3.7
    y2_top = y2_center + lane_width/2   # = 7.4

    # 1) Graue Flächen
    ax.add_patch(Rectangle((x_min, y1_bot),
                        width=x_max-x_min, height=lane_width,
                        facecolor='lightgray', edgecolor='none', zorder=0))
    ax.add_patch(Rectangle((x_min, y2_bot),
                        width=x_max-x_min, height=lane_width,
                        facecolor='lightgray', edgecolor='none', zorder=0))

    # 2) Weiße Randlinien (durchgehend)
    ax.plot([x_min, x_max], [y1_bot, y1_bot],
            color='white', linewidth=2, zorder=1)
    ax.plot([x_min, x_max], [y2_top, y2_top],
            color='white', linewidth=2, zorder=1)

    # 3) Mittellinie (gestrichelt)
    ax.plot([x_min, x_max], [y1_top, y1_top],
            color='white', linewidth=2, linestyle=(0, (10, 10)), zorder=1)

    # --- Fahrzeuge zeichnen ---
    # Ego
    ego_corners = ego_car.get_corners(CAR_LENGTH=4.5, CAR_WIDTH=1.8)
    ax.add_patch(Polygon(ego_corners, closed=True,
                        facecolor='tab:blue', edgecolor='black', alpha=0.5, zorder=2))
    # Optional „Sensorbereich“ um Ego
    ego_outline = ego_car.get_corners(CAR_LENGTH=8, CAR_WIDTH=2.4)
    ax.add_patch(Polygon(ego_outline, closed=False,
                        edgecolor='tab:blue', alpha=0.5, zorder=2))
            # ID-Label im Zentrum des Vierecks
    centroid = np.mean(ego_corners, axis=0)
    ax.text(centroid[0], centroid[1],
            "Ego",
            color='black',
            fontsize=9,
            fontweight='bold',
            ha='center', va='center',
            zorder=3)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    # Humans
    for i, human in enumerate(humans):
        h_corn = human.get_corners(CAR_LENGTH=4.5, CAR_WIDTH=1.8)
        ax.add_patch(Polygon(h_corn, closed=True,
                            facecolor='tab:red', edgecolor='black', alpha=0.5, zorder=2))
        h_outline = human.get_corners(CAR_LENGTH=8, CAR_WIDTH=2.4)
        ax.add_patch(Polygon(h_outline, closed=False, facecolor='tab:red',
                            edgecolor='tab:red', alpha=0.5, zorder=2))
        # ID-Label im Zentrum des Vierecks
        centroid = np.mean(h_corn, axis=0)


    ax.grid(False)  # optional: das Raster über der Straße stört hier



    plt.pause(0.01)

import sys


class Tee:
    """Schreibt alles, was in stdout landet, auch in eine Datei."""
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
    def flush(self):
        for f in self.files:
            f.flush()


import os
for sim in range(1,5, 1):

    sim_dir = os.path.join("experiment7", f"sim_{sim:02d}")
    os.makedirs(sim_dir, exist_ok=True)

    log_path = os.path.join(sim_dir, "log.txt")
    log_file = open(log_path, "w")
    old_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file)

    print(f"--- Simulation #{sim} gestartet ---")
    # Beispiel-Loop:
    fig, ax = plt.subplots()
    fig_history, ax_history = plt.subplots()

    world = World()

    # if sim %2 == 0:
    #     behav = "calm"
    #     world.add_human(x=-2.5, y=Param.lane_centres[1], v=8, behavior=behav)
    # else: 
    #     behav = "aggressive"
    #     world.add_human(x=-2.5, y=Param.lane_centres[1], v=8, behavior=behav)

    
    if sim == 1:
        world.add_human(x=10, y=Param.lane_centres[0], v=1, behavior="aggressive")
        world.add_human(x=-10, y=Param.lane_centres[0], v=1, behavior="aggressive" )
        world.add_human(x=-2.5, y=Param.lane_centres[1], v=1, behavior="aggressive")
        world.add_human(x=8, y=Param.lane_centres[1], v=1, behavior="aggressive")
        world.add_human(x=-10, y=Param.lane_centres[1], v=1, behavior="aggressive")
    elif sim == 2: 
        world.add_human(x=10, y=Param.lane_centres[0], v=1, behavior="aggressive")
        world.add_human(x=-10, y=Param.lane_centres[0], v=1, behavior="aggressive" )
        world.add_human(x=-2.5, y=Param.lane_centres[1], v=1, behavior="aggressive")
        world.add_human(x=8, y=Param.lane_centres[1], v=1, behavior="aggressive")
        world.add_human(x=-10, y=Param.lane_centres[1], v=1, behavior="calm")
    elif sim == 3: 
        world.add_human(x=10, y=Param.lane_centres[0], v=1, behavior="aggressive")
        world.add_human(x=-10, y=Param.lane_centres[0], v=1, behavior="aggressive" )
        world.add_human(x=-2.5, y=Param.lane_centres[1], v=1, behavior="calm")
        world.add_human(x=8, y=Param.lane_centres[1], v=1, behavior="aggressive")
        world.add_human(x=-10, y=Param.lane_centres[1], v=1, behavior="aggressive")
    elif sim == 4: 
        world.add_human(x=10, y=Param.lane_centres[0], v=1, behavior="aggressive")
        world.add_human(x=-10, y=Param.lane_centres[0], v=1, behavior="aggressive" )
        world.add_human(x=-2.5, y=Param.lane_centres[1], v=1, behavior="calm")
        world.add_human(x=8, y=Param.lane_centres[1], v=1, behavior="aggressive")
        world.add_human(x=-10, y=Param.lane_centres[1], v=1, behavior="calm")
    elif sim == 5: 
        world.add_human(x=10, y=Param.lane_centres[0], v=1, behavior="calm")
        world.add_human(x=-10, y=Param.lane_centres[0], v=1, behavior="calm" )
        world.add_human(x=-2.5, y=Param.lane_centres[1], v=1, behavior="calm")
        world.add_human(x=8, y=Param.lane_centres[1], v=1, behavior="calm")
        world.add_human(x=-10, y=Param.lane_centres[1], v=1, behavior="calm")

    
    #drivers in current lane


    for human in world.humans:
        print(f"Human {human.id}: behavior = {human.behavior}")


    oppoID = world.selectOpponent()
    world.oppoID = oppoID

    planner = POMCP(world , oppoID, gamma = 0.8 , c = 0.8)
    world.planner = planner

    # Calculate policy in a loop
    time = 0

    while time <= 50:
        if time%3 == 0:
            plot_cars_history(world.ego, world.humans, ax_history, time *0.5)
            frame_path_h = os.path.join(sim_dir, f"history_{time:03d}.png")
            #frame_path_h = f"experiment3/frame_history.png"
            fig_history.savefig(frame_path_h, dpi=80)

        plot_cars(world.ego, world.humans, ax, time *0.5)
        frame_path = os.path.join(sim_dir, f"frame_{time:03d}.png")
        #frame_path = f"experiment3/frame_{time:03d}.png"
        fig.savefig(frame_path, dpi=80)
        
        #action EGO
        actionEgo = planner.search()
        #action Human:
        observation = []
        actionHumans = []
        for i in range(len(world.humans)):
            actionHuman = world.humans[i].control(world.ego, world.humans )
            actionHumans.append(actionHuman)
            if i in planner.oppoID:
                observation.append(actionHuman)

        #Execute action
        root = planner.tree.nodes[-1]
        world._one_step(actionEgo, actionHumans, particles= root.particles)
        if world.collision(None, []):
            print("collision--------------------------------------------------------------------------------------------")
            break
        # Select Opponent
        oppoID = world.selectOpponent()
        world.oppoID = oppoID
        #Prune + Belief Update + update opponents

        planner.posteriorUpdate(actionEgo, observation, oppoID)
        
        # Print belief 
        root = planner.tree.nodes[-1]
        print_belief_percentages(root.particles)


        print("Real Action taken --------- {}".format(actionEgo))
        
        time += 1

    sys.stdout = old_stdout
    log_file.close()
    print("Simulation {} beendet------------------------------------------------".format(sim))
    plt.close(fig)
    plt.close(fig_history)

