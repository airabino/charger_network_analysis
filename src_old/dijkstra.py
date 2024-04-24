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

default_objectives = [
    {
        'field': 'network_distance',
        'initial': [0],
        'update': lambda x, v: [xi + v for xi in x],
        'cost': lambda x: sum(x) / len(x),
    }
]


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

            if cost[self.range_field][idx] < reset:
                
                cost[self.time_field][idx] += (
                    (reset - cost[self.range_field][idx]) / rate + delay)

                cost[self.price_field][idx] += (
                    (reset - cost[self.range_field][idx]) * price)

                cost[self.range_field][idx] = reset

        return cost

def dijkstra(graph, origins, **kwargs):  
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

    destinations = kwargs.get('destinations', [])
    objectives = kwargs.get('destinations', default_objectives)
    constraints = kwargs.get('destinations', [])

    return_paths = kwargs.get('return_paths', False)

    if return_paths:

        paths = {origin: [origin] for origin in origins}

    else:

        paths = None

    adjacency = graph._adj
    # For speed-up (and works for both directed and undirected graphs)

    path_values = {}  # dictionary of cost values for paths

    path_costs = {}

    visited = {} # dictionary of costs-to-reach for nodes

    destinations_visited = 0

    if len(destinations) == 0:

        destinations_to_visit = maxsize # If no targets are provided then search all nodes

    else:

        destinations_to_visit = len(destinations)
        # If targets are provided then search until all are seen

    # Heap Queue is used for search, efficiently allows for tracking of nodes
    # and pushing/pulling
    c = count() # use the count c to avoid comparing nodes (may not be able to)
    heap = [] # heap is heapq with 3-tuples (cost, c, node)

    for origin in origins:

        visited[origin] = 0 # Source is seen at the start of iteration and at 0 cost

        # Adding the source tuple to the heap (initial cost, count, id)
        values = {}

        for objective in objectives:

            values[objective['field']] = objective['initial']

        for constraint in constraints:

            values[constraint['field']] = constraint['initial']

        heappush(heap, (0, values, next(c), origin))

    while heap: # Iterating while there are accessible unseen nodes

        # Popping the smallest unseen node from the heap

        (cost, values,  _, source) = heappop(heap)

        if source in path_values:

            continue  # already searched this node.

        path_values[source] = values
        path_costs[source] = cost
        # Checking if the current source is a search target
        # If all targets are reached then the search is terminated

        if source in destinations:

            destinations_visited += 1

        if destinations_visited >= destinations_to_visit:

            break

        # Iterating through the current source node's adjacency
        for target, link in adjacency[source].items():

            current = graph._node[target]

            current_values = deepcopy(values)

            feasible = True

            for constraint in constraints:

                # Updating objective for link
                current_values[constraint['field']] = constraint['update'](
                    current_values[constraint['field']], link.get(constraint['field'], 1)
                    )

                # print(current_values)

                # Checking if link traversal is possible
                feasible *= constraint['feasible'](current_values[constraint['field']])

            if not feasible:

                continue

            cost = 0

            for objective in objectives:

                # Updating objective for link
                current_values[objective['field']] = objective['update'](
                    current_values[objective['field']], link.get(objective['field'], 1)
                    )

                # Adding the target node cost
                current_values[objective['field']] = objective['update'](
                    current_values[objective['field']], current.get(objective['field'], 1)
                    )

                # Updating the weighted cost for the path
                cost += objective['cost'](current_values[objective['field']])
                
                # Charging if availabe
                if 'charger' in graph._node[target]:

                    current_values = current['charger'].update(current_values)

            not_visited = target not in visited
            savings = cost < visited.get(target, 0)

            if not_visited or savings:

                visited[target] = cost

                heappush(heap, (cost, current_values, next(c), target))

                if paths is not None:

                    paths[target] = paths[source] + [target]

    return path_costs, path_values, paths