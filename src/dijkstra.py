'''
Module for Dijkstra routing

Implementation is based on NetworkX shortest_paths
'''
import numpy as np

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

        values = values + link.get(self.field, 1)

        return values, values <= self.limit

    def compare(self, values, approximation):

        return values, values < approximation

def dijkstra(graph, origins, **kwargs):
    '''
    Flexible implementation of Dijkstra's algorithm.

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

    destinations = kwargs.get('destinations', [])
    objective = kwargs.get('objective', Objective())
    return_paths = kwargs.get('return_paths', True)

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

            if edge.get('feasible', True):

                node = nodes[target]

                # Updating states for edge traversal
                values_target = objective.update(
                    values, edge,
                    )

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

def multi_directional_dijkstra(graph, origins, **kwargs):
    '''
    Generalized implementation of bi-directional Dijkstra. Finds shortest paths
    between all pairs of origins somewhat quicker than iteratively running
    single-directional Dijkstra (as above) for all origins.

    Please see this helpful guide explaining bi-directional Dijkstra:
    https://www.homepages.ucl.ac.uk/~ucahmto/math/2020/05/30/bidirectional-dijkstra.html

    or NetworkX's implementation:
    https://networkx.org/documentation/stable/reference/algorithms/generated/
    networkx.algorithms.shortest_paths.weighted.bidirectional_dijkstra.html

    Single-directional Dijkstra expands the search from a single origin and finds a path
    from that origin to each destinations. Alternatively, single-directional Dijkstra can
    be used to find a single path to each destination from the closest of a set of origins.
    Multi-directional Dijkstra finds a path from each origin to each origin (and incidental
    paths from origins to non-origin nodes). Iteratively performing single-directional
    Dijkstra requires exploration starting from each origin successively. Muliti-directional
    Dijkstra saves some time by simultaneously searching from all origins and finding
    optimal merging points for paths from the different origins.

    Multi-directional Dijkstra works by storing and updating an estimation of the shortest
    path distance between each origin and each other origin called mu. Mu is updated for a
    pair each time a shorter path is found. If the cmbined radius of the search for a given
    pair is greater than the current approximation of the shortest path between the pair
    then the current shortes path is considered to be the optimal path. If optimal paths
    for all pairs are deemed to have been found then the search is stopped and optimal
    paths are returned.

    Multi-directional Dijkstra is subject to all single-directional Dijkstra faults.
    '''

    objective = kwargs.get('objective', Objective())
    return_paths = kwargs.get('return_paths', False)

    destinations = origins

    infinity = objective.infinity()

    paths = {d: {o: [o] for o in origins} for d in destinations}

    nodes = graph._node
    edges = graph._adj

    # dictionary of costs for paths
    path_values = {o: {o: objective.initial() for o in origins} for o in origins} 

    path_costs = {o: {} for o in origins} # dictionary of objective values for paths

    visited = {o: {} for o in origins} # dictionary of costs-to-reach for nodes

    # Estimation of origin-pair shortest paths
    mu = {d: {o: np.inf for o in origins} for d in destinations}

    # Status of search for origin-pair shortest paths - True for self paths
    connections = {d: {o: o == d for o in origins} for d in destinations}

    # Values of the origin-pair paths
    connection_values = {d: {o: objective.initial() for o in origins} for d in destinations}

    connection_paths = {d: {o: [o] for o in origins} for d in destinations}

    c = count() # use the count c to avoid comparing nodes (may not be able to)
    heaps = {o: [] for o in origins} # heap is heapq with 3-tuples (cost, c, node)

    for origin in origins:

        # Source is seen at the start of iteration and at 0 cost
        visited[origin][origin] = objective.initial()

        # Adding the source tuple to the heap (initial cost, count, id)
        values = objective.initial()

        heappush(heaps[origin], (0, next(c), values, origin))

    # Iterating while there are accessible unseen nodes
    while any([heap for heap in heaps.values()]):


        sources = {o: o for o in origins}

        for origin, heap in heaps.items():

            # Popping the lowest cost unseen node from the heap
            if not heap:

                continue

            cost, _, values, source = heappop(heap)

            sources[origin] = source

            # Relaxation of source
            if source in path_costs[origin]:

                continue  # already searched this node.

            path_costs[origin][source] = cost

            # Iterating through the current source node's adjacency
            for target, edge in edges[source].items():

                node = nodes[target]

                # Updating states for edge traversal
                values_target, feasible = objective.update(
                    values, edge, node,
                    )

                if feasible:

                    # Updating the weighted cost for the path
                    cost, savings = objective.compare(
                        values_target, visited[origin].get(target, infinity)
                        )

                    edge_cost = cost - path_costs[origin][source]

                    if savings:
                       
                        visited[origin][target] = values_target
                        path_values[origin][target] = values_target

                        heappush(heap, (cost, next(c), values_target, target))

                        paths[origin][target] = paths[origin][source] + [target]

                    # Updating origin-pair paths
                    for destination in origins:

                        if destination != origin:

                            if target in path_costs[destination]:

                                tentative_mu = (
                                    path_costs[origin][source] +
                                    edge_cost+
                                    path_costs[destination][target]
                                    )

                                tentative_path = (
                                    paths[origin][source] +
                                    paths[destination][target][::-1]
                                    )

                                tentative_values, feasible = objective.combine(
                                    path_values[origin][target],
                                    path_values[destination][target]
                                    )

                                # Updating estimation if tentative path is shorter
                                # than current path. Also update path and values.
                                if (tentative_mu < mu[origin][destination]) and feasible:

                                    mu[origin][destination] = tentative_mu

                                    connection_paths[origin][destination] = tentative_path

                                    connection_values[origin][destination] = (
                                        tentative_values
                                        )

        # Updating origin-pair search status
        complete = True

        for origin in sources.keys():

            for destination in sources.keys():

                # Combined search radius
                search_radius = (
                    path_costs[origin][sources[origin]] +
                    path_costs[destination][sources[destination]]
                    )

                # If combined search radius is greater than current estimate end search
                if search_radius >= mu[origin][destination]:

                    path_costs[origin][destination] = mu[origin][destination]

                    connections[origin][destination] = True

                complete *= connections[origin][destination]

        # If shortest-paths found for all origin-pairs end iteration
        if complete:

            break

    return path_costs, connection_values, connection_paths