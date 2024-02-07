'''
Module for Dijkstra routing

Code is based on (is an edited version of):
NetworkX shortest_paths.weighted._dijkstra_multisource

Edits are to allow for native tracking of multiple shortest path simultaneously.
For example, one could get a shortest path weighted by 'distance' but also
want to know path 'time', this edited code allows for this to be done efficiently.
'''
from heapq import heappop, heappush
from itertools import count
from sys import float_info, maxsize

class Charger():

    def __init__(self, node, fields, amounts, probability = 1):

        self.node = node
        self.fields = fields
        self.amounts = amounts
        self.probability = probability

    def update(self, capacities, probability):

        for idx, field in enumerate(self.fields):

            capacities[idx] = self.amounts[idx]

        return capacities, probability * self.probability

class Objective():

    def __init__(self, fields, limits, initial_values = None, weights = None):

        self.fields = fields
        self.limits = limits

        self.n = len(fields)

        self.initial_values = initial_values

        if initial_values is None:

            self.initial_values = [0] * self.n

        self.initial_capacities = [self.limits[idx] - self.initial_values[idx] \
            for idx in range(self.n)]

        self.weights = weights

        if weights is None:

            self.weights = [1] + [0] * (self.n - 1)

    def cost(self, link):

        return [link[field] for field in self.fields]

    def update(self, link, values, capacities):

        cost = self.cost(link)

        new_values = [values[idx] + cost[idx] for idx in range(self.n)]
        new_capacities = [capacities[idx] - cost[idx] for idx in range(self.n)]
        feasible = all([new_capacities[idx] >= 0 for idx in range(self.n)])

        return new_values, new_capacities, feasible

    def weighted_sum(self, values):

        weighted_values = sum(
            [self.weights[idx] * values[idx] for idx in range(self.n)]
            )

        return weighted_values

def dijkstra(graph, sources, objective, targets = [], chargers = {}, **kwargs):
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
    minimum_path_probability = kwargs.get('minimum_path_probability', 0)

    charger_nodes = list(chargers.keys())

    if return_paths:

        paths = {source: [source] for source in sources}

    else:

        paths = None

    graph_succ = graph._adj
    # For speed-up (and works for both directed and undirected graphs)

    path_values = {}  # dictionary of final distances
    path_capacities = {}
    path_probability = {}
    seen = {}

    targets_visited = 0

    if len(targets) == 0:

        targets_to_hit = maxsize

    else:

        targets_to_hit = len(targets)

    # fringe is heapq with 3-tuples (distance,c,node)
    # use the count c to avoid comparing nodes (may not be able to)

    c = count()
    fringe = []

    for source in sources:

        seen[source] = 0
        heappush(
            fringe,
            ((objective.initial_values, objective.initial_capacities, 1), next(c), source)
            )

    while fringe:

        (d,  _, v) = heappop(fringe)

        if v in path_values:

            continue  # already searched this node.

        path_values[v] = d[0]
        path_capacities[v] = d[1]
        path_probability[v] = d[2]

        if v in targets:

            targets_visited += 1

        if targets_visited >= targets_to_hit:

            break

        for u, e in graph_succ[v].items():

            tentative_values, tentative_capacities, feasible = objective.update(
                e, path_values[v], path_capacities[v])

            tentative_probability = d[2]

            if not feasible:

                continue

            if u in charger_nodes:

                tentative_capacities, tentative_probability = chargers[u].update(
                    tentative_capacities, path_probability[v])

            # print(tentative_probability)

            if tentative_probability < minimum_path_probability:

                continue

            weighted_tentative_values = objective.weighted_sum(tentative_values)

            if u not in seen or weighted_tentative_values < seen[u]:

                seen[u] = weighted_tentative_values

                heappush(
                    fringe,
                    (
                        (tentative_values, tentative_capacities, tentative_probability),
                        next(c),
                        u
                        )
                    )

                if paths is not None:

                    paths[u] = paths[v] + [u]

    return path_values, paths


def dijkstra_old(graph, sources, weights, targets = [], chargers = {}, **kwargs):
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

    weights : dictionary - {field: cutoff}
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

    charger_nodes = list(chargers.keys())

    if return_paths:

        paths = {source: [source] for source in sources}

    else:

        paths = None

    n_weights=len(weights)

    for weight, limit in weights.items():

        if limit <= 0:

            weights[weight] = float_info.max

    graph_succ = graph._adj
    # For speed-up (and works for both directed and undirected graphs)

    path_weights = {}  # dictionary of final distances
    path_capacities = {}
    seen = {}

    targets_visited = 0

    if len(targets) == 0:

        targets_to_hit = maxsize

    else:

        targets_to_hit = len(targets)

    # fringe is heapq with 3-tuples (distance,c,node)
    # use the count c to avoid comparing nodes (may not be able to)

    c = count()
    fringe = []

    for source in sources:

        seen[source] = 0
        heappush(fringe, (([0,]*n_weights, weights.copy()), next(c), source))

    while fringe:

        (d,  _, v) = heappop(fringe)

        if v in path_weights:

            continue  # already searched this node.

        path_weights[v] = d[0]
        path_capacities[v] = d[1]

        if v in targets:

            targets_visited += 1

        if targets_visited >= targets_to_hit:

            break

        for u, e in graph_succ[v].items():

            cost = [e.get(field, 1) for field in weights.keys()]

            if cost[0] is None:

                continue

            vu_weights = [path_weights[v][idx] + cost[idx] for idx in range(n_weights)]
            vu_capacities = {k: path_capacities[v][k] - cost[idx] \
                for idx, k in enumerate(weights.keys())}

            cutoff_exceeded = any([vu_capacities[k] < 0 \
                for k in weights.keys()])

            if cutoff_exceeded:

                continue

            if u in charger_nodes:

                for key, val in chargers[u].items():

                    vu_capacities[key] = val

            if u in path_weights:

                u_dist = path_weights[u]

                if vu_weights[0] < u_dist[0]:

                    raise ValueError("Contradictory paths found:", "negative weights?")

            elif u not in seen or vu_weights[0] < seen[u]:

                seen[u] = vu_weights[0]

                heappush(fringe, ((vu_weights, vu_capacities), next(c), u))

                if paths is not None:

                    paths[u] = paths[v] + [u]

    return path_weights, paths