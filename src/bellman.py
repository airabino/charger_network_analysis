'''
Module for Dijkstra routing

Code is based on (is an edited version of):
NetworkX shortest_paths.weighted._dijkstra_multisource

Edits are to allow for native tracking of multiple shortest path simultaneously.
For example, one could get a shortest path weighted by 'distance' but also
want to know path 'time', this edited code allows for this to be done efficiently.
'''
import time
import numpy as np

from copy import deepcopy
from heapq import heappop, heappush
from itertools import count
from sys import float_info, maxsize

class Charger():

    def __init__(self, reset, rate, delay, price, **kwargs):

        self.reset = reset
        self.rate = rate
        self.delay = delay
        self.price = price

        self.range_field = kwargs.get('range_field', 'range')
        self.time_field = kwargs.get('time_field', 'time')
        self.price_field = kwargs.get('price_field', 'price')
        self.rng = np.random.default_rng(kwargs.get('seed', None))

    def get_random_state(self):

        rn = self.rng.random()

        return self.reset(rn), self.rate(rn), self.delay(rn), self.price(rn)

    def update(self, cost):
        # print('ccccc')

        for idx in range(len(cost[self.range_field])):

            reset, rate, delay, price = self.get_random_state()
            # print(reset, rate, delay, price)

            if cost[self.range_field][idx] > reset:
                
                cost[self.time_field][idx] += (
                    (cost[self.range_field][idx] - reset) / rate + delay)

                cost[self.price_field][idx] += (
                    (cost[self.range_field][idx] - reset) * price)

                cost[self.range_field][idx] = reset

        return cost

class Objective():

    def __init__(self, field, initial, feasible, weight):

        self.field = field
        self.initial = initial
        self.feasible = feasible
        self.weight = weight

        self.n = len(self.initial)

    def update(self, cost, entity):

        # print('a', self.field, cost, entity)
        # print(entity.get(self.field, 0))

        entity_value = entity.get(self.field, 0)
        # print('c', entity_value)

        for idx in range(self.n):

            cost[self.field][idx] += entity_value

        # print('b', self.field, cost)

        return cost

def bellman(graph, destinations, objectives, n = 1, origins = [], **kwargs):
    """Uses Dijkstra's algorithm to find shortest weighted paths

    Code is based on (is an edited version of):
    NetworkX shortest_paths.weighted._dijkstra_multisource

    Edits are to allow for native tracking of multiple shortest path simultaneously.

    Parameters
    ----------
    graph : NetworkX graph

    sources : non-empty iterable of nodes
        Starting nodes for paths. If this is just an iterable containing
        a single node, then all paths computed by this function will
        start from that node. If there are two or more nodes in this
        iterable, the computed paths may begin from any one of the start
        nodes.

    objectives : dictionary - {field: {'limit': limit, 'weight': weight}}
        Cumulative values for path fields will be returned - if any cutoff is exceeded
        in reaching a node the node is considered unreachable via the given path.
        AT LEAST ONE FIELD IS REQUIRED.

    targets : iterable of nodes - optionally empty
        Ending nodes for path. Search is halted when all targets are reached. If empty
        all nodes will be reached if possible.

    chargers : Dictionary - {node: weights}
        Dictionary of nodes that reset given weights to zere. As an example
        a charger can be a node which resets a distance weight to zero allowing for
        further distances to be reached.

    return_paths : Boolean
        Boolean whether or not to compute paths dictionary. If False None
        is returned for the paths output. COMPUTING PATHS WILL INCREASE RUN-TIME.

    Returns
    -------
    path_weights : dictionary
        Path weights from source nodes to target nodes.

    paths : dictionary
        Dictionary containing ordered lists of nodes passed on shortest
        path between the origin node and other nodes. If return_paths == False
        then None will be returned.
    """

    return_paths = kwargs.get('return_paths', False)

    if return_paths:

        paths = {destination: [destination] for destination in destinations}

    else:

        paths = None

    if tpye(graph) is nx.classes.digraph.DiGraph:

        graph = graph.reverse(default = True)

    adjacency = graph._adj
    # For speed-up (and works for both directed and undirected graphs)

    path_values = {}  # dictionary of cost values for paths

    visited = {} # dictionary of costs-to-reach for nodes

    destinations_visited = 0

    if len(destinations) == 0:

        originss_to_visit = maxsize # If no targets are provided then search all nodes

    else:

        originss_to_visit = len(origins)
        # If targets are provided then search until all are seen

    # Heap Queue is used for search, efficiently allows for tracking of nodes
    # and pushing/pulling
    c = count() # use the count c to avoid comparing nodes (may not be able to)
    heap = [] # heap is heapq with 3-tuples (cost, c, node)

    # tc = 0

    for destination in destinations:

        visited[origin] = 0 # Source is seen at the start of iteration and at 0 cost
        # Adding the source tuple to the heap (initial cost, count, id)
        cost = {objective.field: objective.final for objective in objectives}
        heappush(heap, (0, cost, next(c), origin))

    while heap: # Iterating while there are accessible un-visited nodes

        # Popping the smallest unseen node from the heap

        (wc, cost,  _, target) = heappop(heap)

        if target in path_values:

            continue  # already searched this node.

        path_values[target] = cost
        # Checking if the current source is a search target
        # If all targets are reached then the search is terminated

        if target in origins:

            origins_visited += 1

        if origins_visited >= origins_to_visit:

            break

        # Iterating through the current source node's adjacency
        for source, link in adjacency[target].items():

            tentative_cost = deepcopy(cost)

            feasible = True
            weighted_cost = 0

            # print('a', tentative_cost)

            for objective in objectives:
                # print(objective.__dict__)

                # t0 = time.time()
                # Updating objective for link
                tentative_cost = objective.update(tentative_cost, link)


                # Checking if link traversal is possible
                feasible *= objective.feasible(tentative_cost)


                # Adding the target node cost
                tentative_cost = objective.update(tentative_cost, graph._node[source])


                # Updating the weighted cost for the path
                weighted_cost += objective.weight(tentative_cost)

                
                # Charging if availabe
                if 'charger' in graph._node[source]:

                    # print('dddddd', graph._node[target])

                    tentative_cost = graph._node[source]['charger'].update(tentative_cost)

                # tc += time.time() - t0
                


            # print('b', tentative_cost)

            not_visited = source not in visited
            savings = weighted_cost < visited.get(source, 0)
            # print((not_visited or savings) and feasible)

            if (not_visited or savings) and feasible:

                # print(tentative_cost)

                visited[source] = weighted_cost

                heappush(heap, (weighted_cost, tentative_cost, next(c), source))

                if paths is not None:

                    paths[source] = [source] + paths[target]

    # print('tc', tc)

    return path_values, paths