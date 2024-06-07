'''
Module for Bellman routing

Implementation is based on NetworkX shortest_paths
'''
import numpy as np

from collections import deque
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

        values += link.get(self.field, 1)

        return values, values <= self.limit

    def compare(self, values, approximation):

        return values, values < approximation

def bellman(graph, origins, **kwargs):
    '''
    Flexible implementation of Bellman's algorithm.

    Depends on an Objective object which contains the following four functions:

    values = initial() - Function which produces the starting values of each problem state
    to be applied to the origin node(s)

    values = infinity() - Function which produces the starting values for each non-origin
    node. The values should be intialized such that they are at least higher than any
    conceivable value which could be attained during routing.

    values, feasible = update(values, edge, node) - Function which takes current path state
    values and updates them based on the edge traversed and the target node and whether the
    proposed edge traversal is feasible. This function returns the values argument and a
    boolean feasible.

    values, savings = compare(values, approximation) - Function for comparing path state
    values with the existing best approximation at the target node. This function returns
    the values argument and a boolean savings.
    '''

    destinations = kwargs.get('destinations', None)
    objective = kwargs.get('objective', Objective())
    heuristic = kwargs.get('heuristic', True)
    return_paths = kwargs.get('return_paths', False)

    predecessor = {target: [] for target in origins}

    values = {target: objective.initial() for target in origins}
    cost = {target: 0 for target in origins}

    # Heuristic Storage setup. Note: use None because nodes cannot be None
    nonexistent_edge = (None, None)
    predecessor_edge = {origin: None for origin in origins}
    recent_update = {origin: nonexistent_edge for origin in origins}

    adjacency = graph._adj
    nodes = graph._node
    infinity = objective.infinity()
    n = len(graph)

    count = {}
    queue = deque(origins)
    in_queue = set(origins)

    # print(cost, values, 'a')

    aaa = 0

    while queue:
        aaa +=  1
        # print(aaa, end = '\r')

        # print('b')

        source = queue.popleft()
        in_queue.remove(source)

        # Skip relaxations if any of the predecessors of source is in the queue.
        if all(pred_source not in in_queue for pred_source in predecessor[source]):

            values_source = values[source]

            for target, edge in adjacency[source].items():

                values_target, feasible = objective.update(values_source, edge, nodes[target])
                
                if feasible:

                    cost_target, savings = objective.compare(
                        values_target, values.get(target, infinity)
                        )

                    if savings:
                        # In this conditional branch we are updating the path with target.
                        # If it happens that some earlier update also added node target
                        # that implies the existence of a negative cycle since
                        # after the update node target would lie on the update path twice.
                        # The update path is stored up to one of the source nodes,
                        # therefore source is always in the dict recent_update
                        if heuristic:

                            if target in recent_update[source]:

                                print('Negative cycle found!', target, recent_update[target])
                                predecessor[target].append(source)

                                break

                            # Transfer the recent update info from source to target if the
                            # same source node is the head of the update path.
                            # If the source node is responsible for the cost update,
                            # then clear the history and use it instead.
                            if (
                                (target in predecessor_edge) and
                                (predecessor_edge[target] == source)
                                ):

                                recent_update[target] = recent_update[source]

                            else:

                                recent_update[target] = (source, target)

                        if target not in in_queue:

                            queue.append(target)
                            in_queue.add(target)

                            count_target = count.get(target, 0) + 1

                            if count_target == n:

                                print('Negative cycle found!')
                                break

                            count[target] = count_target

                            # print('c')

                        values[target] = values_target
                        cost[target] = cost_target
                        predecessor[target] = [source]
                        predecessor_edge[target] = source

                    elif cost.get(target) is not None and cost_target == cost.get(target):

                        predecessor[target].append(source)

    # print(cost, values, 'b')

    if return_paths:

        paths = {}

        origins = set(origins)

        destinations = destinations if destinations is not None else predecessor

        for destination in destinations:

            path_generator = paths_from_predecessors(
                origins, destination, predecessor
                )

            paths[destination] = next(path_generator)

    else:

        paths = None

    # print(cost, values, 'a')

    return cost, values, paths

def paths_from_predecessors(origins, destination, predecessor):

    seen = {destination}

    stack = [[destination, 0]]

    top = 0

    while top >= 0:

        node, i = stack[top]

        if node in origins:

            yield [p for p, n in reversed(stack[: top + 1])]

        if len(predecessor[node]) > i:

            stack[top][1] = i + 1
            successor = predecessor[node][i]

            if successor in seen:

                continue

            else:

                seen.add(successor)

            top += 1

            if top == len(stack):

                stack.append([successor, 0])

            else:

                stack[top][:] = [successor, 0]

        else:

            seen.discard(node)
            top -= 1