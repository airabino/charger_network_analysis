import time

import numpy as np
import networkx as nx

from heapq import heappop, heappush
from itertools import count
from sys import maxsize

from numba import jit

class Objective():

    def __init__(self, field = 'weight', edge_limit = np.inf, path_limit = np.inf):

        self.field = field
        self.edge_limit = edge_limit
        self.path_limit = path_limit

    def initial(self):

        return 0

    def infinity(self):

        return np.inf

    def zero(self):

        return {self.field: 0}

    def update(self, values, link):

        edge_value = link.get(self.field, 1)

        values += edge_value

        return values, (values <= self.path_limit) and (edge_value <= self.edge_limit)

    def compare(self, values, approximation):

        return values, values < approximation

    def cost(self, values):

        cost = values.get(self.field, 1)

        if cost <= self.edge_limit:

            return cost

        else:

            return self.infinity()

    def combine(self, values_0, values_1):

        new_path_values = {k: values_0.get(k, 0) + values_1.get(k, 0) for k in values_0.keys()}

        cost_0 = values_0.get(self.field, 1)
        cost_1 = values_1.get(self.field, 1)

        # print(cost_0, cost_1)

        new_path_cost = cost_0 + cost_1

        new_path_feasible = new_path_cost <= self.path_limit

        return new_path_cost, new_path_values, new_path_feasible


def floyd_warshall(graph, fields, **kwargs):

    adjacency = {f: nx.to_numpy_array(graph, weight = f) for f in fields}
    adjacency_primary = adjacency[fields[0]]

    n = len(adjacency_primary)

    origins = kwargs.get('origins', list(range(n)))
    destinations = kwargs.get('destinations', list(range(n)))
    pivots = kwargs.get('pivots', list(range(n)))
    tolerance = kwargs.get('tolerance', 0)
    limit = kwargs.get('limit', np.inf)

    if not pivots:

        pivots = list(range(n))

    if tolerance == 0:

        costs = np.zeros_like(adjacency_primary)
        predecessors = np.zeros_like(adjacency_primary, dtype = int)

        costs, predecessors = _floyd_warshall(
            adjacency_primary,
            pivots,
            costs,
            predecessors,
        )

        paths = {}
        values = {}

        for origin in origins:

            paths[origin] = {}
            values[origin] = {}

            for destination in destinations:

                path = recover_path(
                    predecessors, origin, destination
                    )

                paths[origin][destination] = path

                values[origin][destination] = (
                    {f: recover_path_costs(adjacency[f], path) for f in fields}
                    )

    else:

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

        paths = {}
        values = {}

        for origin in origins:

            paths[origin] = {}
            values[origin] = {}

            for destination in destinations:

                path = recover_paths(
                    extended, origin, [[destination]], []
                    )

                paths[origin][destination] = path

                values[origin][destination] = [
                    {f: recover_path_costs(adjacency[f], p) for f in fields} for p in path
                    ]

    return costs, values, paths

def recover_path(predecessors, origin, destination):

    max_iterations = len(predecessors)

    path = [destination]

    idx = 0

    while (origin != destination) and (idx <= max_iterations):

        destination = predecessors[origin][destination]
        path = [destination] + path

        idx +=1

    return path

def recover_path_costs(adjacency, path):

    cost = 0

    for idx in range(len(path) - 1):
        # print()

        cost += adjacency[path[idx]][path[idx + 1]]

    return cost

@jit(nopython = True, cache = True)
def _floyd_warshall(adjacency, pivots, costs, predecessors):

    n = len(adjacency)

    for source in range(n):
        for target in range(n):

            costs[source][target] = adjacency[source][target]
            predecessors[source][target] = source

    for pivot in pivots:
        for source in range(n):
            for target in range(n):

                if costs[source][pivot] + costs[pivot][target] < costs[source][target]:

                    costs[source][target] = costs[source][pivot] + costs[pivot][target]
                    predecessors[source][target] = predecessors[pivot][target]

    return costs, predecessors

@jit(nopython = True, cache = True)
def _floyd_warshall_multi(adjacency, pivots, costs, predecessors, tolerance = .05):

    tolerance += 1.

    n = len(adjacency)

    store = []

    for source in range(n):

        for target in range(n):

            costs[source][target] = adjacency[source][target]
            predecessors[source][target] = source

    for pivot in pivots:
        for source in range(n):
            for target in range(n):

                costs_new = costs[source][pivot] + costs[pivot][target]

                if costs_new < costs[source][target]:

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


def recover_tree(predecessors, origin, destinations, tree):

    if len(destinations) == 0:

        return tree
    
    destination = destinations.pop()

    # If the origin is the destination then move to next branch in queue
    if origin == destination:

        return recover_tree(predecessors, origin, destinations, tree)

    # If the origin is not the destination then add edges to tree and queue
    else:

        for predecessor in predecessors[origin][destination]:

            tree.append((predecessor, destination))
            destinations.append(predecessor)

        return recover_tree(predecessors, origin, destinations, tree)

def recover_paths(predecessors, origin, paths, complete_paths):

    if len(paths) == 0:

        return complete_paths
    
    path = paths.pop()
    destination = path[0]

    # If the origin is the destination then move to next branch in queue
    if origin == destination:

        complete_paths.append(path)

        return recover_paths(predecessors, origin, paths, complete_paths)

    # If the origin is not the destination then add new paths and update queue
    else:

        for predecessor in predecessors[origin][destination]:

            paths.append([predecessor] + path)

        return recover_paths(predecessors, origin, paths, complete_paths)