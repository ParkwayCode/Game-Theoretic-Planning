import math
import numpy as np
from numpy.random import choice
from numpy.random import uniform
from POMDP.tree import *
from collections import Counter
from scipy.spatial import KDTree
import random







class POMCP:
    def __init__(self,  world, oppoID,  gamma=0.9, c = 0.9,
                 timeout=1000, max_particles=300, horizon = 5):
                # timeout=15000, max_particles=3000, horizon = 10):

        if not (0 < gamma < 1):
            raise ValueError("gamma must be in (0,1)")

        self.world = world
        self.gamma = gamma
        self.timeout = timeout
        self.max_particles = max_particles
        self.horizon = horizon
        self.numberOpponents = len(world.humans)
        self.gain = 1
        self.c = c

        self.epsilon = 0.5

        self.states = world.mdp.STATES
        self.actions = world.mdp.ACTIONS
        self.Q = world.mdp.Q

        self.oppoID = oppoID


        self.tree = SearchTree()
        self.KD = KDTree(self.states)

        self.tree.nodes[-1].egoCar   = world.ego
        self.tree.nodes[-1].humans   = world.humans

        self.initBelief()

    def initBelief(self):
    
        k_values = [0, 1, 2]
        #k_probs = [0.3, 0.3, 0.4]            #Probability of k values
        lambda_range = (0.5, 1.0)            #Range of Lambda Values
        #lambda_values = [0.5, 1, 3]
        lambda_values = [1, 3, 7]
        # Fill the particles list with the initial belief
        particles = []
        combinations = [(k, lam) for k in k_values for lam in lambda_values]
        num_combinations = len(combinations)

        repeats_per_combination = self.max_particles // num_combinations
    

        for combo in combinations:
            particles.extend([combo] * repeats_per_combination)

        

        root = self.tree.nodes[-1]
        root.particles = [
            particles.copy() 
            for _ in range(self.numberOpponents)
        ]



    def search(self):

        #Get the root Node
        root = self.tree.nodes[-1]  
        for _ in range(self.timeout):
            root.hiddenState = []
            for i in range(self.numberOpponents):
                idx = choice(len(root.particles[i]))
                root.hiddenState.append(root.particles[i][idx])


            self.rollout( node_id=-1, depth=0)

        # get best action
        action, _ = self.bestAction(node_id=-1, use_ucb=False)

        return action
    
    

    def bestAction(self, node_id, use_ucb=True):

        node = self.tree.nodes[node_id]
        best_score = -math.inf
        best_action = None
        best_child = None

        for action, child_id in node.children.items():
            child = self.tree.nodes[child_id]

            # unvisited nodes first
            if use_ucb and child.visits == 0:
                return action, child_id

            # get score
            if use_ucb:
                score = UCB(node.visits, child.visits, child.value, self.c)
            else:
                score = child.value

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child_id
    
        return best_action, best_child






    def rollout(self,  node_id, depth):
        
        if depth == self.horizon:
            return 0
        # Check significance of update
        if (self.gamma**depth < self.epsilon or self.gamma == 0 ) and depth != 0:
            return 0

        node = self.tree.nodes[node_id]

        #Expansion
        if node.visits == 0:

            for actionEGO in self.world.legal_actions():

                nextID = self.tree.expand(node_id, actionEGO, is_action=True)

                humanActions, _, observation = self.ComputeHumanPolicy(node, node.hiddenState)
                 # Get the observation Node and expand if it does not exist
                childID, _ = self.tree.getObservationNode(nextID, observation)
                child = self.tree.nodes[childID]

                self.simulate(actionEGO, humanActions, node, child)
                
        #Selection
        actionEGO, nextID = self.bestAction(node_id, use_ucb=True)
        
        humanActions, _, observation = self.ComputeHumanPolicy(node, node.hiddenState)
        # Get the observation Node and expand if it does not exist
        childID, newNode = self.tree.getObservationNode(nextID, observation)
        child = self.tree.nodes[childID]

        if newNode:
            self.simulate(actionEGO, humanActions, node, child)
        #Belief Update
        self.updateChild(child, node)
        



        cum_reward = 0
        cum_reward += child.R + self.gamma*self.rollout( childID, depth + 1)

        #Backpropagation
        node.visits += 1
        actionNode = self.tree.nodes[nextID]
        actionNode.visits += 1
        actionNode.value += (cum_reward -  actionNode.value)/ actionNode.visits

        return cum_reward
    

    def updateChild(self, child, node):
        child.hiddenState = node.hiddenState
        #Update Belief
        for i in range(self.numberOpponents):
            #only update the belief of the real opponents
            if i in self.oppoID:
                child.particles[i].append(child.hiddenState[i])
                if len(child.particles[i]) > self.max_particles:
                    child.particles[i].pop(0)
            
        #information gain
        info_gain = entropy(node.particles) - entropy(child.particles)
        #Update reward
        child.R +=  self.gain * info_gain
    


    def simulate(self, action_EGO, action_HUMAN, node, child):

        #TODO: implement Simulator
        #State: [dx_R dy_R dx_H dv_R dv_H dtheta_R gamma]:
        egoCar = copy.deepcopy(node.egoCar)
        humans = copy.deepcopy(node.humans)

        #nextPositions is the observation
        child.egoCar, child.humans , child.R = self.world._one_step(
                                                        action_EGO, action_HUMAN, egoCar, humans, node.particles) 

        #init childs particles
        child.particles = copy.deepcopy(node.particles)
        self.updateChild(child, node)



        



    #TODO: Update Policy to belief
    def ComputeHumanPolicy(self, node, belief):
        
        humanActions = []
        humanProbabilities = []
        observation = []

        egoCar = node.egoCar
        tree = self.KD

        
        neg_actions = np.array([-2, -1])
        pos_actions = np.array([0, 1, 2])
        
       

        for i, human in enumerate(node.humans):
            
            #default action
            action_i = 0
            prob_i = 1.0
            if i in self.oppoID:
                k, lamda = belief[i]

                if egoCar.egoMDP.indicating:
                    if k == 0:
                        p_yield = 1.0 / len(self.actions)
                    elif k == 1:
                        p_yield = (lamda + 1) / (lamda + 2.0)
                    else:  # k == 2
                        p_yield = 1.0 / (lamda + 1.0)
                    p_yield = max(0.1, min(1.0 - 0.1, p_yield)) #restrict yield to [0, 1]
                    prob_neg = p_yield / len(neg_actions)
                    prob_pos = (1.0 - p_yield) / len(pos_actions)

                    probs = {}
                    for a in neg_actions:
                        probs[a] = prob_neg
                    for a in pos_actions:
                        probs[a] = prob_pos

                    
                    acts = list(probs.keys())
                    ps = np.array([probs[a] for a in acts])
                    ps /= ps.sum() 
                    chosen = np.random.choice(acts, p=ps.tolist())
                    action_i = int(chosen)
                    prob_i = float(probs[action_i])

                    observation.append(action_i)
                else:

                    dx = human.x -egoCar.x 
                    dy = human.y -egoCar.y
                    state = dx, dy, human.v, egoCar.v, egoCar.theta

                    if k == 0:
                        prob_i = 1.0 / len(self.actions)
                        action_i = np.random.choice(self.actions)
                    else:
                        #find index of the next state
                        _, idx = tree.query(state)

                        Qk = self.Q[k - 1]
                        row = Qk[idx]
                        row_max = row.max()
                        logits  = lamda * (row - row_max)
                        exp_row = np.exp(logits)
                        pi      = exp_row / exp_row.sum()

                        action_idx = np.random.choice(len(self.actions), p= pi)
                        prob_i = pi[action_idx]
                        action_i = self.actions[action_idx]
                    observation.append(action_i)

            humanActions.append(action_i)
            humanProbabilities.append(prob_i)


        return humanActions, humanProbabilities, observation

  


    def posteriorUpdate(self, actionEgo, observation, oppoID):
        
        self.tree.prune_after_action(actionEgo,observation)

        # prior = self.tree.nodes[-1].particles.copy()
        # new_particles = []
        # new_particles = [[] for _ in range(self.numberOpponents)]
        # full = False
        # for i in range(self.numberOpponents):
        #     if i not in self.oppoID:
        #         new_particles[i] = prior[i]
        
        # while True:
        #     if len(self.oppoID) == 0:
        #         break
        #     belief = []
        #     for i in range(self.numberOpponents):
        #         #if i in self.oppoID:
        #         idx = choice(len(prior[i]))
        #         particle = prior[i][idx]
        #         belief.append(particle)

        #     _, _, o_next = self.ComputeHumanPolicy(self.tree.nodes[-1], belief)

        #     if o_next == observation:
        #         for i in range(self.numberOpponents):
        #             if i in self.oppoID:
        #                 new_particles[i].append(belief[i])
        #                 if len(new_particles[i]) == self.max_particles:
        #                     full = True
        #     if full:
        #         break
       

        # self.tree.nodes[-1].particles = new_particles

        self.oppoID = oppoID


    # def posteriorUpdate(self, actionEgo, observation, oppoID):
    #     """
    #     Aktuelles Root-Belief wird anhand der echten Aktion (actionEgo)
    #     und der real beobachteten Opponent-Aktionen (observation: List[int])
    #     mit Gewichts-Resampling aktualisiert.
    #     """
    #     # 1) Zuerst prunen wir den Baum wie gehabt
    #     self.tree.prune_after_action(actionEgo, observation)

    #     # 2) Kopiere die alte Partikelmenge (Liste pro Agent)
    #     root = self.tree.nodes[-1]
    #     prior_particles = [root.particles[i].copy() for i in range(self.numberOpponents)]
        
    #     # 3) Wir wollen für jede Opponent-ID einen neuen Satz Partikel mit Gewichtung
    #     new_particles = [[] for _ in range(self.numberOpponents)]

    #     if len(oppoID) == 0:
    #         # Kein Gegner → nichts tun
    #         for i in range(self.numberOpponents):
    #             new_particles[i] = prior_particles[i]
    #         root.particles = new_particles
    #         self.oppoID = oppoID
    #         return

    #     # 4) Für jedes Partikel (für jeden Opponent) berechnen wir die Likelihood,
    #     #    d.h. P(o_real | s, θ, a_ego). Da wir mehrere "echte" Beobachtungen
    #     #    (z.B. actionHuman1, actionHuman2, …) haben, multiplizieren wir die
    #     #    Einzelwahrscheinlichkeiten über alle Agenten, die in oppoID sind.

    #     # Extrahiere den aktuellen physikalischen Zustand aus root.egoCar/root.humans
    #     # Dieser bleibt gleich für alle Partikel beim Update
    #     current_root_node = self.tree.nodes[-1]

    #     # Für jedes Opponent i in oppoID erstellen wir eine Liste von Gewichten
    #     weights_per_agent = {i: [] for i in oppoID}

    #     # Und wir brauchen eine gemeinsame Partikel-Liste-Länge (max_particles)
    #     N = self.max_particles

    #     # Jetzt durchlaufen wir für jeden i∈oppoID alle alten Partikel prior_particles[i]
    #     # und berechnen die Likelihood gegen die real beobachtete Aktion observation[j],
    #     # wobei j den Index von i in oppoID ist.
    #     for idx_agent in oppoID:
    #         obs_real = observation[oppoID.index(idx_agent)]
    #         weights = []
    #         for part in prior_particles[idx_agent]:
    #             # Erzeuge eine temporäre Belief‐Liste, in der alle Nicht‐Gegner None sind,
    #             # aber bei Gegner idx_agent setzen wir part.
    #             temp_belief = [None] * self.numberOpponents
    #             temp_belief[idx_agent] = part
    #             # Für alle anderen Gegner (falls es mehr als einen gibt), müssen wir
    #             # irgendetwas einsetzen – wir setzen sie hier ebenfalls zufällig aus dem prior,
    #             # damit die ComputeHumanPolicy‐Wahrscheinlichkeiten korrekt sind.
    #             for j in oppoID:
    #                 if j != idx_agent:
    #                     temp_belief[j] = random.choice(prior_particles[j])

    #             # Berechne die Wahrscheinlichkeiten aller Human-Aktionen unter dem
    #             # kombinierten Belief temp_belief
    #             _, probs, _ = self.ComputeHumanPolicy(current_root_node, temp_belief)

    #             # P(o_real | θ_i, s, a_ego) ist das entsprechende Element in probs
    #             weight_i = probs[idx_agent]
    #             weights.append(weight_i)
    #         # Falls alle Gewichte 0 sind, ersetzen wir sie durch 1 (uniform)
    #         W = np.array(weights, dtype=float)
    #         if W.sum() == 0:
    #             W[:] = 1.0
    #         W /= W.sum()
    #         weights_per_agent[idx_agent] = W.tolist()

    #     # 5) Jetzt führen wir für jeden Gegner idx_agent ein gewichts-basiertes Resampling
    #     #    durch, d.h. ziehen N Partikel aus prior_particles[idx_agent] mit Wahrscheinlichkeit W.
    #     for idx_agent in oppoID:
    #         W = weights_per_agent[idx_agent]
    #         # random.choices zieht mit Gewichten W N Partikel (mit Zurücklegen)
    #         new_particles[idx_agent] = random.choices(prior_particles[idx_agent], weights=W, k=N)

    #     # 6) Alle Nicht-Gegner behalten ihre ursprünglichen Partikel (keine Änderung):
    #     for i in range(self.numberOpponents):
    #         if i not in oppoID:
    #             new_particles[i] = prior_particles[i].copy()

    #     # 7) Setze den neuen Root-Belief
    #     root.particles = new_particles
    #     self.oppoID = oppoID




def entropy(particles):
    #Calculates the Shannon Entropy based on the belief state
    k_vals = [p[0] for p in particles]
    counts = Counter(k_vals)
    total  = len(k_vals)

    #   H = - sum p * log2(p)
    H = 0.0
    for cnt in counts.values():
        p = cnt / total
        H -= p * np.log2(p)
    return H






