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
from itertools import count

from .progress_bar import ProgressBar

from .dijkstra import dijkstra
from .bellman import bellman
from .floyd_warshall import floyd_warshall
from .graph import graph_from_nlg

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

        self.fields = kwargs.get('fields', ['time', 'distance', 'price'])
        self.weights = kwargs.get('weights', [1, 0, 0])
        self.limits = kwargs.get('limits', [np.inf, np.inf, np.inf])
        self.n = len(self.fields)

    def initial(self):

        return {field: 0 for field in self.fields}

    def infinity(self):

        return {self.fields[idx]: np.inf for idx in range(self.n)}

    def update(self, values, link):

        feasible = True

        values_new = {}

        for idx in range(self.n):

            values_new[self.fields[idx]] = (
                values[self.fields[idx]] + link.get(self.fields[idx], 0)
                )

            feasible *= values_new[self.fields[idx]] <= self.limits[idx]

        # print('bbb', values_new)

        return values_new, feasible

    def compare(self, values, comparison):

        cost_new = 0
        cost_current = 0

        for idx in range(self.n):

            # print(values, self.fields[idx])

            cost_new += values[self.fields[idx]] * self.weights[idx]
            cost_current += comparison[self.fields[idx]] * self.weights[idx]

        savings = (cost_new < cost_current) or np.isnan(cost_current)

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

        cost, values, paths = algorithm(
            atlas,
            [origin_atlas],
            objective = objective,
            )

        adj = {}

        destinations_reached = np.intersect1d(
            list(values.keys()),
            destinations_atlas,
            )

        for destination in destinations_reached:

            nodes = atlas_to_graph[destination]

            for node in nodes:

                adj[node] = values[destination]

        graph._adj[origin] = adj

    return graph

def reduction(atlas, graph, objective = Graph_From_Atlas(), algorithm = dijkstra):
    '''
    Adds adjacency to graph by routing on atlas
    '''

    graph_to_atlas, atlas_to_graph = node_assignment(graph, atlas)

    destinations = list(graph.nodes)

    destinations_atlas = [graph_to_atlas[node] for node in destinations]

    intersections = []

    for source, adj in atlas._adj.items():

        if len(adj) > 2:

            intersections.append(source)

    reduced_nodes = np.unique(destinations_atlas + intersections)
    reduced_node_ids = {node: idx for idx, node in enumerate(reduced_nodes)}
    # print(reduced_node_ids)

    _node = atlas._node

    nodes = []
    links = []

    null_edge = objective.infinity()

    for origin in ProgressBar(reduced_nodes):

        node = _node[origin]
        node['id'] = reduced_node_ids[origin]

        nodes.append(node)

        # origin_atlas = graph_to_atlas[origin]

        cost, values, paths = dijkstra(
            atlas,
            [origin],
            destinations = reduced_nodes,
            objective = objective,
            terminate_at_destinations = True
            )


        destinations_reached = np.intersect1d(
            list(values.keys()),
            reduced_nodes,
            )


        for destination in destinations_reached:

            link = values.get(destination, null_edge)
            link['source'] = reduced_node_ids[origin]
            link['target'] = reduced_node_ids[destination]

            links.append(link)

    return graph_from_nlg({'nodes': nodes, 'links': links})

def adjacency_fw(atlas, graph, fields = ['time']):
    '''
    Adds adjacency to graph by routing on atlas
    '''

    graph_to_atlas, atlas_to_graph = node_assignment(graph, atlas)

    destinations = list(graph.nodes)

    destinations_atlas = [graph_to_atlas[node] for node in destinations]

    costs, values, paths = floyd_warshall(
        atlas, fields,
        origins = destinations_atlas,
        destinations = destinations_atlas,
        )

    for origin in ProgressBar(destinations):

        # origin_atlas = graph_to_atlas[origin]

        # cost, values, paths = algorithm(
        #     atlas,
        #     [origin_atlas],
        #     objective = objective,
        #     )

        # adj = {}

        # destinations_reached = np.intersect1d(
        #     list(values.keys()),
        #     destinations_atlas,
        #     )

        for destination in destinations:

            nodes = atlas_to_graph[destination]

            for node in nodes:

                adj[node] = values[origin][destination]

        graph._adj[origin] = adj

    return graph