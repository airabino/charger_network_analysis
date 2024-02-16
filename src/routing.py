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

        for idx in range(len(cost[self.range_field])):

            reset, rate, delay, price = self.get_random_state()

            if cost[self.range_field][idx] < reset:
                
                cost[self.time_field][idx] += (
                    (reset - cost[self.range_field][idx]) / rate + delay)

                cost[self.price_field][idx] += (
                    (reset - cost[self.range_field][idx]) * price)

                cost[self.range_field][idx] = reset

        return cost

def dijkstra(graph, origins, **kwargs):  
    """
    Uses Dijkstra's algorithm to find weighted shortest paths

    Code is based on NetworkX shortest_paths.weighted._dijkstra_multisource

    This implementation of Dijkstra's method is designed for high flexibility
    at some cost to efficiency. Specifically, this function implements Stochastic
    Cost with Risk Allowance Minimization Dijkstra (SCRAM-D) routing. As such
    this function allows for an optimal route to be computed for probabilistic
    node/link costs by tracking N scenarios in parallel with randomly sampled
    costs and minimizing the expectation of cost subject to constraints which
    may also be based on cost expectation. Additionally, nodes amy contain Charger
    objects which serve to reset a given state to a pre-determined value and
    may effect other states.

    Example - Battery Electric Vehicle (BEV) routing:

    graph - Graph or DiGraph containing a home location, several destinations,
    and M chargers of varying reliability and links for all objects less than
    300 km apart.

    origins - [home]

    destinations - [Yellowstone Park, Yosemite Park, Grand Canyon]

    states - {
        'distance': {
            'field': 'distance',
            'initial': [0] * n_cases,
            'update': lambda x, v: [xi + v for xi in x],
            'cost': lambda x: 0,
        },
        'price': {
            'field': 'price',
            'initial': [0] * n_cases,
            'update': lambda x, v: [xi + v for xi in x],
            'cost': lambda x: 0,
        },
        'time': {
            'field': 'time',
            'initial': [0] * n_cases,
            'update': lambda x, v: [xi + v for xi in x],
            'cost': lambda x: src.utilities.super_quantile(x, risk_tolerance),
        },
    }

    constraints - {
        'range': {
            'field': 'range',
            'initial': [vehicle_range] * n_cases,
            'update': lambda x, v: [xi + v for xi in x],
            'feasible': lambda x: src.utilities.super_quantile(x, risk_tolerance) > min_range,
        },
    }

    Parameters
    ----------
    graph: a NetworkX Graph or DiGraph

    origins: non-empty iterable of nodes
        Starting nodes for paths. If origins is an iterable containing
        a single node, then all paths computed by this function will
        start from that node. If there are two or more nodes in origins
        the shortest path to a given destination may begin from any one
        of the start nodes.

    destinations: iterable of nodes - optionally empty
        Ending nodes for path. If destinations is not empty then search
        continues until all reachable destinations are visited. If destinations
        is empty then the search continues until all reachable nodes are visited.

    states: dictionary with the below fields:
        'field': The relevant node/link property for integration
        'initial': Initial values for integration
        'update': Function which updates path values for nodes/links
        'cost': Function for computing a cost expectation from values

    constraints: dictionary with the below fields:
        'field': The relevant node/link property for integration
        'initial': Initial values for integration
        'update': Function which updates path values for nodes/links
        'feasible': Function which returns a Boolean for feasibility from values

    return_paths: Boolean
        Boolean whether or not to compute paths dictionary. If False None
        is returned for the paths output. COMPUTING PATHS WILL INCREASE RUN-TIME.

    Returns
    -------

    path_costs : dictionary
        Path cost expectations.

    path_values : dictionary
        Path objective values.

    paths : dictionary
        Dictionary containing ordered lists of nodes passed on shortest
        path between the origin node and other nodes. If return_paths == False
        then None will be returned.
    """

    destinations = kwargs.get('destinations', [])
    states = kwargs.get('states', default_objectives)
    constraints = kwargs.get('constraints', [])
    return_paths = kwargs.get('return_paths', False)

    if return_paths:

        paths = {origin: [origin] for origin in origins}

    else:

        paths = None

    adjacency = graph._adj

    path_values = {}  # dictionary of costs for paths

    path_costs = {} # dictionary of objective values for paths

    visited = {} # dictionary of costs-to-reach for nodes

    destinations_visited = 0

    if len(destinations) == 0:

        # If no destinations are provided then search all nodes
        destinations_to_visit = maxsize

    else:

        #If destinations are provided then search until all are seen
        destinations_to_visit = len(destinations)

    c = count() # use the count c to avoid comparing nodes (may not be able to)
    heap = [] # heap is heapq with 3-tuples (cost, c, node)

    for origin in origins:

        visited[origin] = 0 # Source is seen at the start of iteration and at 0 cost

        # Adding the source tuple to the heap (initial cost, count, id)
        values = {}

        for key, info in states.items():

            values[key] = info['initial']

        for key, info in constraints.items():

            values[key] = info['initial']

        heappush(heap, (0, values, next(c), origin))

    while heap: # Iterating while there are accessible unseen nodes

        # Popping the lowest cost unseen node from the heap
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

            for key, info in constraints.items():

                # Updating objective for link
                current_values[key] = info['update'](
                    current_values[key], link.get(info['field'], 1)
                    )

                # Checking if link traversal is possible
                feasible *= info['feasible'](current_values[key])

            if not feasible:

                continue

            cost = 0

            for key, info in states.items():

                # Updating objective for link
                current_values[key] = info['update'](
                    current_values[key], link.get(info['field'], 1)
                    )

                # Adding the target node cost
                current_values[key] = info['update'](
                    current_values[key], current.get(info['field'], 1)
                    )

                # Updating the weighted cost for the path
                cost += info['cost'](current_values[key])
                
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