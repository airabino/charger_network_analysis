import time

import numpy as np
import networkx as nx

from heapq import heappop, heappush
from itertools import count
from sys import maxsize

from numba import jit


def floyd_warshall(graph, fields, **kwargs):
    '''
    Implements the Floyd-Warshall algorithm for all-pairs routing

    args:

    graph is a NetworkX Graph
    fields is a list of edge attributes - the first one listed will be used for routing

    kwargs:

    origins - list of nodes in graph from which routes will start
    destinations - list of nodes in graph at which reoutes will end
    pivots - list of nodes in graph which can serve as intermediaries in routes

    if origins, destinations, or pivots are not provided then all nodes will be used

    tolerance - float threshold of disambiguation for selecting alterante paths

    if a non-zero tolerance is provided then alternate paths may be produced
    '''

    # Creating adjacency matrices
    adjacency = {f: nx.to_numpy_array(graph, weight = f, nonedge = np.inf) for f in fields}
    adjacency_primary = adjacency[fields[0]]

    node_to_idx = {k: idx for idx, k in enumerate(graph.nodes)}
    idx_to_node = {idx: k for idx, k in enumerate(graph.nodes)}

    n = len(adjacency_primary)

    # Processing kwargs
    origins = kwargs.get('origins', list(graph.nodes))
    destinations = kwargs.get('destinations', list(graph.nodes))
    pivots = kwargs.get('pivots', list(graph.nodes))
    tolerance = kwargs.get('tolerance', 0)

    # origins_idx = [node_to_idx[k] for k in origins]
    # destinations_idx = [node_to_idx[k] for k in destinations]
    pivots_idx = [node_to_idx[k] for k in pivots]

    
    if tolerance == 0: # Only store optimal routes

        # Running the Floyd Warshall algorithm
        costs = np.zeros_like(adjacency_primary)
        predecessors = np.zeros_like(adjacency_primary, dtype = int)


        costs, predecessors = _floyd_warshall(
            adjacency_primary,
            pivots_idx,
            costs,
            predecessors,
        )


        # Recovering paths and values
        paths = {}
        values = {}

        for origin in origins:

            paths[origin] = {}
            values[origin] = {}

            for destination in destinations:

                path = recover_path(
                    predecessors, node_to_idx[origin], node_to_idx[destination]
                    )

                paths[origin][destination] = [idx_to_node[n] for n in path]

                values[origin][destination] = (
                    {f: recover_path_costs(adjacency[f], path) for f in fields}
                    )

    else: # Search for alternate routes within threshold of disambiguation

        # Running the Floyd Warshall algorithm
        costs = np.zeros_like(adjacency_primary)
        predecessors = np.zeros_like(adjacency_primary, dtype = int)

        costs, predecessors, store = _floyd_warshall_multi(
            adjacency_primary,
            pivots,
            costs,
            predecessors,
            tolerance = tolerance,
        )

        extended = extended_predecessors(
            costs, predecessors, store, tolerance = tolerance * 10
        )

        # Recovering paths and values
        paths = {}
        values = {}

        for origin in origins:

            paths[origin] = {}
            values[origin] = {}

            for destination in destinations:

                path = recover_paths(
                    extended, origin, destination
                    )

                paths[origin][destination] = path

                values[origin][destination] = [
                    {f: recover_path_costs(adjacency[f], p) for f in fields} for p in path
                    ]

    return costs, values, paths

def recover_path(predecessors, origin, destination):
    '''
    recovers paths by working backward from destination to origin
    '''

    max_iterations = len(predecessors)

    path = [destination]

    idx = 0

    while (origin != destination) and (idx <= max_iterations):

        destination = predecessors[origin][destination]
        path = [destination] + path

        idx +=1

    return path

def recover_path_costs(adjacency, path):
    '''
    Recovers costs for a path on an adjacency matrix
    '''

    cost = 0

    for idx in range(len(path) - 1):

        cost += adjacency[path[idx]][path[idx + 1]]

    return cost

@jit(nopython = True, cache = True)
def _floyd_warshall(adjacency, pivots, costs, predecessors):
    '''
    Implementation of Floyd Warshall algorithm
    https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm
    '''

    n = len(adjacency)

    # Creating initial approximations
    for source in range(n):
        for target in range(n):

            # Initial assumption is that source is the direct predecessor to target
            # and that the cost is adjacency[source][target]. Non-edges should be
            # set to infinite cost for the algorithm to produce correct results
            costs[source][target] = adjacency[source][target]
            predecessors[source][target] = source

    # Updating approximations
    for pivot in pivots:
        for source in range(n):
            for target in range(n):

                # if source-pivot-target is lower cost than source-target then update
                if costs[source][pivot] + costs[pivot][target] < costs[source][target]:

                    costs[source][target] = costs[source][pivot] + costs[pivot][target]
                    predecessors[source][target] = predecessors[pivot][target]

    return costs, predecessors

@jit(nopython = True, cache = True)
def _floyd_warshall_multi(adjacency, pivots, costs, predecessors, tolerance = .05):
    '''
    Implementation of Floyd Warshall algorithm
    https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm

    This implementation stores some sub-optimal approximations in order to
    produce alternate paths
    '''

    tolerance += 1.

    n = len(adjacency)

    store = [] # List for storing non-optimal approximations

    # Creating initial approximations
    for source in range(n):
        for target in range(n):

            # Initial assumption is that source is the direct predecessor to target
            # and that the cost is adjacency[source][target]. Non-edges should be
            # set to infinite cost for the algorithm to produce correct results
            costs[source][target] = adjacency[source][target]
            predecessors[source][target] = source

    # Updating approximations
    for pivot in pivots:
        for source in range(n):
            for target in range(n):

                costs_new = costs[source][pivot] + costs[pivot][target]

                # if source-pivot-target is lower cost than source-target then update
                if costs_new < costs[source][target]:

                    # If the difference is less than the threshold of disambiguation
                    # then store the previous approximation
                    if costs[source][target] < min([tolerance * costs_new, np.inf]):

                        store.append(
                            (
                                source, target,
                                predecessors[source][target],
                                costs[source][target],
                                )
                            )

                    costs[source][target] = costs[source][pivot] + costs[pivot][target]
                    predecessors[source][target] = predecessors[pivot][target]

    return costs, predecessors, store

def extended_predecessors(costs, predecessors, store, tolerance = .05):
    '''
    Computes a multi-predecessors dictionary to allow for alternative routing
    '''

    tolerance += 1.

    n = len(costs)

    extended = {}

    for source in range(n):

        extended[source] = {}

        for target in range(n):

            extended[source][target] = {predecessors[source][target]}

    for predecessor in store:

        s, t, p, c = predecessor

        if c <= tolerance * costs[s][t]:

            extended[s][t].add(p)

    return extended

def recover_paths(predecessors, origin, destination):
    '''
    Recovers multiple branching path alternatives
    '''

    paths = []

    heap = []

    c = count()

    heappush(heap, (next(c), [destination]))

    while heap:

        _, path = heappop(heap)

        destinations = predecessors[origin][path[0]]

        for destination in destinations:

            if destination == origin:

                paths.append([destination] + path)

            else:

                heappush(heap, (c, [destination] + path))

    return paths