'''
Module for computing adjacency for a graph via routing on another graph. An example
would be computing travel times between cities connected by highways or latency between
computers connected via the internet. Another case would be compting network distances
between all points in a subset of a greater network. In any case, the nodes of the former
network must be coincident, or nearly so, with nodes in the latter network.

In this module the graph for which adjacency is being computed will be referred to as the
"graph" while the graph on which the routing occurs will be referred to as the "atlas". In
cases where either could be used "graph" will be used as default.
'''

import numpy as np

from sys import maxsize

from scipy.spatial import KDTree

from .progress_bar import ProgressBar

from .dijkstra import dijkstra
from .bellman import bellman

# Routing functions and related objects

def closest_nodes_from_coordinates(graph, x, y):
    '''
    Creates an assignment dictionary mapping between points and closest nodes
    '''

    # Pulling coordinates from graph
    xy_graph = np.array([(n['x'], n['y']) for n in graph._node.values()])
    xy_graph = xy_graph.reshape((-1,2))

    # Creating spatial KDTree for assignment
    kd_tree = KDTree(xy_graph)

    # Shaping input coordinates
    xy_query = np.vstack((x, y)).T

    # Computing assignment
    result = kd_tree.query(xy_query)

    node_assignment = []

    for idx in range(len(x)):

        node = result[1][idx]

        node_assignment.append({
            'id':node,
            'query':xy_query[idx],
            'result':xy_graph[node],
            })

    return node_assignment

def node_assignment(graph, atlas):

    x, y = np.array(
        [[val['x'], val['y']] for key, val in graph._node.items()]
        ).T

    graph_nodes = np.array(
        [key for key, val in graph._node.items()]
        ).T

    atlas_nodes = closest_nodes_from_coordinates(atlas, x, y)

    graph_to_atlas = (
        {graph_nodes[idx]: atlas_nodes[idx]['id'] for idx in range(len(graph_nodes))}
        )
    
    atlas_to_graph = {}

    for key, val in graph_to_atlas.items():

        if val in atlas_to_graph.keys():

            atlas_to_graph[val] += [key]

        else:

            atlas_to_graph[val] = [key]

    return graph_to_atlas, atlas_to_graph

class Graph_From_Atlas():

    def __init__(self, **kwargs):

        self.fields = kwargs.get('fields', ['time', 'distance', 'pice'])
        self.weights = kwargs.get('weights', [1, 0, 0])
        self.limits = kwargs.get('limits', [np.inf, np.inf, np.inf])
        self.n = len(self.fields)

    def initial(self):

        return {field: 0 for field in self.fields}

    def infinity(self):

        return {self.fields[idx]: np.inf for idx in range(self.n)}

    def update(self, values, link, node):
        # print('a')

        feasible = True

        for idx in range(self.n):

            values[self.fields[idx]] += link.get(self.fields[idx], 0)

            feasible *= values[self.fields[idx]] <= self.limits[idx]

        # print(feasible)

        return values, feasible

    def compare(self, values, comparison):

        cost_new = 0
        cost_current = 0

        for idx in range(self.n):

            cost_new += values[self.fields[idx]] * self.weights[idx]
            cost_current += comparison[self.fields[idx]] * self.weights[idx]

        savings = (cost_new < cost_current) or np.isnan(cost_current)

        # print(cost_new, cost_current, savings)

        return cost_new, savings

def adjacency(atlas, graph, objective = Graph_From_Atlas(), algorithm = dijkstra):
    '''
    Adds adjacency to graph by routing on atlas
    '''

    graph_to_atlas, atlas_to_graph = node_assignment(graph, atlas)

    destinations = list(graph.nodes)

    destinations_atlas = [graph_to_atlas[node] for node in destinations]

    for origin in ProgressBar(destinations):

        origin_atlas = graph_to_atlas[origin]

        # destinations_atlas.remove(origin_atlas)

        # print(destinations_atlas)

        cost, values, paths = algorithm(
            atlas,
            [origin_atlas],
            objective = objective,
            # destinations = destinations_atlas
            )

        adj = {}

        for destination in destinations_atlas:

            nodes = atlas_to_graph[destination]

            for node in nodes:

                adj[node] = values[destination]

        graph._adj[origin] = adj

    return graph








# def adjacency(atlas, graph, weights, **kwargs):
#     '''
#     Computing adjacency for graph by routing along atlas
#     '''

#     kwargs.setdefault('pb_kwargs', {'disp': True})
#     kwargs.setdefault('dijkstra_kwargs', {'return_paths': False})
#     kwargs.setdefault('compute_all', False)
#     kwargs.setdefault('node_assignment_function', node_assignment)

#     # Maps closest nodes from atlas to graph and graph to atlas
#     graph_to_atlas, atlas_to_graph = kwargs['node_assignment_function'](graph, atlas)

#     # All nodes of graph are assumed to be targets
#     targets = [graph_to_atlas[n] for n in list(graph.nodes)]

#     # Collecting status of all nodes in graph
#     statuses = np.array([n.get('status', 0) for n in graph._node.values()])

#     # Creating sources based on statuses and compute_all
#     if kwargs['compute_all']:

#         sources = targets[:]

#     else:

#         sources = [targets[idx] for idx, status in enumerate(statuses) if status == 0]

#     for n in list(graph.nodes):

#         graph._node[n]['status'] = 1

#     # Computing routes between selected sources and all targets
#     results = multiple_source_dijkstra(atlas, sources, targets, weights, **kwargs)

#     # Compiling edge information from results into 3-tuple for adding to graph
#     edges = []

#     i = 0

#     for result in results:

#         sources = atlas_to_graph[result.pop('source')]
#         targets = atlas_to_graph[result.pop('target')]

#         for source in sources:
#             for target in targets:

#                 edges.append((source, target, result))

#     # Adding edges to graph
#     graph.add_edges_from(edges)

#     return graph