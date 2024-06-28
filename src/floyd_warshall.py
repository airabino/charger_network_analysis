import time

import numpy as np

from heapq import heappop, heappush
from itertools import count
from sys import maxsize

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


def floyd_warshall(graph, origins, **kwargs):

    pivots = kwargs.get('pivots', [k for k in graph.nodes])
    objective = kwargs.get('objective', Objective())
    return_paths = kwargs.get('return_paths', True)

    infinity = objective.infinity()

    if return_paths:

        paths = {origin: [origin] for origin in origins}

    else:

        paths = None

    _node = graph._node
    _adj = graph._adj

    values = {k: {} for k in _node.keys()}

    costs = {k: {} for k in _node.keys()}

    previous = {k: {} for k in _node.keys()}

    for source, adj in _adj.items():

        # previous[source][source] = source

        for target, edge in adj.items():

            values[source][target] = (
                _adj.get(source, {}).get(target, objective.infinity())
                )

            costs[source][target] = objective.cost(values[source][target])

            previous[source][target] = source
            # if costs[source][target] < objective.infinity():

            #     previous[source][target] = source

            # else:

            #     previous[source][target] = None

    # print(costs)

    for pivot in pivots:
        for source in graph.nodes:
            for target in graph.nodes:
                if source != target:
                
                    new_path_cost, new_path_values, new_path_feasible = objective.combine(
                        values[source][pivot], values[pivot][target]
                        )

                    
                    # time.sleep(.001) 

                    _, new_path_savings = objective.compare(
                        new_path_cost, costs[source][target]
                        )

                    if new_path_feasible:
                        
                        if new_path_savings:

                            # print(pivot, source, target, values[source][pivot], values[pivot][target], new_path_cost)

                            # print(new_path_cost)

                            costs[source][target] = new_path_cost
                            values[source][target] = new_path_values
                            previous[source][target] = previous[pivot][target]
                            # previous[source][target] = pivot

    return costs, values, previous








    visited = {} # dictionary of costs-to-reach for nodes


    destinations_visited = 0

    terminals = []

    if terminate_at_destinations:

        terminals = [d for d in destinations if d not in origins]

    c = count() # use the count c to avoid comparing nodes (may not be able to)
    heap = [] # heap is heapq with 3-tuples (cost, c, node)

    for origin in origins:

        # Source is seen at the start of iteration and at 0 cost
        visited[origin] = objective.initial()

        # Adding the source tuple to the heap (initial cost, count, id)
        values = objective.initial()

        heappush(heap, (0, next(c), values, origin))

    while heap: # Iterating while there are accessible unseen nodes

        # Popping the lowest cost unseen node from the heap
        cost, _, values, source = heappop(heap)

        if source in path_values:

            continue  # already searched this node.

        path_values[source] = values
        path_costs[source] = cost

        if source in terminals:

            continue

        for target, edge in edges[source].items():

            if edge.get('feasible', True):

                # Updating states for edge traversal
                values_target, path_feasible = objective.update(
                    values, edge,
                    )

                if path_feasible:

                    # Updating the weighted cost for the path
                    cost, savings = objective.compare(
                        values_target, visited.get(target, infinity)
                        )

                    if savings:
                       
                        visited[target] = values_target

                        heappush(heap, (cost, next(c), values_target, target))

                        if paths is not None:

                            paths[target] = paths[source] + [target]

    return path_costs, path_values, paths