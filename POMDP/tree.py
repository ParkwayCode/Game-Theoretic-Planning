import math
import copy


# Knoten des Suchbaums
class Node:
    def __init__(self, parent, children=None, visits=0, value=0.0, particles=None, x = 0):



        self.parent = parent
        self.children = children if children is not None else {}
        self.visits = visits
        self.value = value
        self.particles = particles if particles is not None else []

        #list of particles [pos_Ego, pos_Human, k, λ] (Default: [])
        if particles is not None:
            self.particles = particles.copy() if isinstance(particles, list) else list(particles)
        else:
            self.particles = []

        self.R = 0
        
        self.egoCar   = 0
        self.humans = 0
        self.hiddenState = []




class SearchTree:
    def __init__(self, init_params=None):
        """
        Initialisiert den Baum mit einem Root-Knoten.
        Standardmäßig wird der Root-Knoten mit ['isRoot', {}, 0, 0, []] angelegt.
          - parent: 'isRoot'
          - children: leeres Dictionary
          - visits: 0
          - value: 0.0
          - particles: leere Liste
        """
        if init_params is None:
            init_params = ['isRoot', {}, 0, 0, []]
        self.count = -1  # Start-ID; Root-Knoten wird unter -1 gespeichert.
        self.nodes = {}
        # Erzeuge den Root-Knoten
        self.nodes[self.count] = Node(
            parent=init_params[0],
            
            particles=copy.deepcopy(init_params[4])
        )

    def expand(self, parent, index, is_action=False):


        self.count += 1  # Neue Knoten-ID
        if is_action:
            new_particles = None
        else:
            new_particles = []
            #Simulate
        new_node = Node(parent=parent, children={}, visits=0, value=0.0, particles=new_particles)
        self.nodes[self.count] = new_node
        # Füge den neuen Knoten im Kinder-Dictionary des Elternknotens unter 'index' hinzu.
        self.nodes[parent].children[index] = self.count

        return self.count




    def isLeaf(self, node_id):

        return self.nodes[node_id].visits == 0

    def getObservationNode(self, h, sample_observation):

        newNode = False
        obs = tuple(sample_observation)
        if obs not in self.nodes[h].children:
            self.expand(parent=h, index=obs, is_action=False)
            newNode = True
        return self.nodes[h].children[obs], newNode


    def prune(self, node_id):

        children_ids = list(self.nodes[node_id].children.values())
        del self.nodes[node_id]
        for child_id in children_ids:
            self.prune(child_id)

    def newRoot(self, new_root):

        self.nodes[-1] = copy.deepcopy(self.nodes[new_root])
        del self.nodes[new_root]
        self.nodes[-1].parent = 'isRoot'
        for child_id in self.nodes[-1].children.values():
            self.nodes[child_id].parent = -1

    def prune_after_action(self, action, observation):
        
        if action not in self.nodes[-1].children:
            raise ValueError("Kein Aktionsknoten für Aktion {} im Root gefunden.".format(action))
        action_node = self.nodes[-1].children[action]
        new_root, _ = self.getObservationNode(action_node, observation)
        obs = tuple(observation)
        if obs in self.nodes[action_node].children:
            del self.nodes[action_node].children[obs]
        self.prune(-1)

        self.newRoot(new_root)


def UCB(total_visits, node_visits, value, c=1.0):

    return value + c * math.sqrt(math.log(total_visits) / node_visits)



