'''
Module for Dijkstra routing

Implementation is based on NetworkX shortest_paths
'''
import numpy as np

from copy import deepcopy
from heapq import heappop, heappush
from itertools import count
from sys import maxsize

class Objective():

    def __init__(self, field = 'weight', limit = np.inf):

        self.field = field
        self.limit = limit

    def initial(self):

        return 0

    def infinity(self):

        return np.inf

    def update(self, values, link, node):

        values += link.get(self.field, 1) + node.get(self.field, 0)

        return values, values <= self.limit

    def compare(self, values, comparison):

        return values, values < comparison

def dijkstra(graph, origins, **kwargs):
    '''
    Flexible implementation of Dijkstra's algorithm
    '''

    destinations = kwargs.get('destinations', [])
    objective = kwargs.get('objective', Objective())
    return_paths = kwargs.get('return_paths', False)

    infinity = objective.infinity()

    if return_paths:

        paths = {origin: [origin] for origin in origins}

    else:

        paths = None

    nodes = graph._node
    edges = graph._adj

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

        # Source is seen at the start of iteration and at 0 cost
        visited[origin] = objective.initial()

        # Adding the source tuple to the heap (initial cost, count, id)
        values = {}

        values = objective.initial()

        heappush(heap, (0, next(c), values, origin))

    while heap: # Iterating while there are accessible unseen nodes

        # Popping the lowest cost unseen node from the heap
        (cost, _, values, source) = heappop(heap)

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
        for target, edge in edges[source].items():

            node = nodes[target]

            current_values = deepcopy(values)

            # Updating states for edge traversal
            current_values, feasible = objective.update(
                current_values, edge, node,
                )

            if feasible:

                # Updating the weighted cost for the path
                cost, savings = objective.compare(
                    current_values, visited.get(target, infinity)
                    )

                if savings:
                   
                    visited[target] = current_values

                    heappush(heap, (cost, next(c), current_values, target))

                    if paths is not None:

                        paths[target] = paths[source] + [target]

    return path_costs, path_values, paths