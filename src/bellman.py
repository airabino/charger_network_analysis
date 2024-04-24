'''
Module for Dijkstra routing

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

        values += link.get(self.field, 1) + node.get(self.field, 0)

        return values, values <= self.limit

    def compare(self, values, comparison):

        return values, values < comparison

def bellman(graph, origins, **kwargs):
    """Calls relaxation loop for Bellman–Ford algorithm and builds paths

    This is an implementation of the SPFA variant.
    See https://en.wikipedia.org/wiki/Shortest_Path_Faster_Algorithm

    Parameters
    ----------
    graph : NetworkX graph

    source: list
        List of source nodes. The shortest path from any of the source
        nodes will be found if multiple origins are provided.

    weight : function
        The weight of an edge is the value returned by the function. The
        function must accept exactly three positional arguments: the two
        endpoints of an edge and the dictionary of edge attributes for
        that edge. The function must return a number.

    pred: dict of lists, optional (default=None)
        dict to store a list of predecessors keyed by that node
        If None, predecessors are not stored

    paths: dict, optional (default=None)
        dict to store the path list from source to each node, keyed by node
        If None, paths are not stored

    dist: dict, optional (default=None)
        dict to store distance from source to the keyed node
        If None, returned dist dict contents default to 0 for every node in the
        source list

    target: node label, optional
        Ending node for path. Path lengths to other destinations may (and
        probably will) be incorrect.

    heuristic : bool
        Determines whether to use a heuristic to early detect negative
        cycles at a hopefully negligible cost.

    Returns
    -------
    dist : dict
        Returns a dict keyed by node to the distance from the source.
        Dicts for paths and pred are in the mutated input dicts by those names.

    Raises
    ------
    NodeNotFound
        If any of `source` is not in `graph`.

    NetworkXUnbounded
        If the (di)graph contains a negative (di)cycle, the
        algorithm raises an exception to indicate the presence of the
        negative (di)cycle.  Note: any negative weight edge in an
        undirected graph is a negative cycle
    """

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

    while queue:

        source = queue.popleft()
        in_queue.remove(source)

        # Skip relaxations if any of the predecessors of source is in the queue.
        if all(pred_source not in in_queue for pred_source in predecessor[source]):

            values_source = values[source]

            for target, edge in adjacency[source].items():

                # cost_target = cost_source + adjacency[source][target][weight]
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

                                # Negative cycle found!
                                predecessor[target].append(source)

                                return target

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

                                # Negative cycle found!
                                return target

                            count[target] = count_target

                        values[target] = values_target
                        cost[target] = cost_target
                        predecessor[target] = [source]
                        predecessor_edge[target] = source

                    elif values.get(target) is not None and values_target == values.get(target):

                        predecessor[target].append(source)

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