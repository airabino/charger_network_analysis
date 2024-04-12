import numpy as np
import networkx as nx

from .graph import graph_from_json, subgraph

def load_graphs():

    atlas = graph_from_json('Outputs/atlas.json')
    graph = graph_from_json('Outputs/graph.json')
    cities = graph_from_json('cities.json')

    return atlas, graph, cities

def tesla_network(graph):

    keep = []

    for source, node in graph._node.items():

        if 'station' not in str(source):

            keep.append(source)

        elif node.get('network', '') == 'Tesla':

            keep.append(source)

    return subgraph(graph, keep)

def non_proprietary_network(graph):

    keep = []

    for source, node in graph._node.items():

        if 'station' not in source:

            keep.append(source)

        elif node.get('network', '') not in ['Tesla' 'RIVIAN_ADVENTURE', 'Non-Neworked']:

            if node.get('corridor', 0):

                keep.append(source)

    return subgraph(graph, keep)

def combined_network(graph):

    keep = []

    for source, node in graph._node.items():

        if 'station' not in source:

            keep.append(source)

        elif node.get('network', '') in ['Tesla', 'eVgo Network', 'Electrify America', 'ChargePoint Network']:

            # if node.get('corridor', 0):

            keep.append(source)

    return subgraph(graph, keep)

def icev_network(graph):

    keep = []

    for source, node in graph._node.items():

        if 'station' not in source:

            keep.append(source)

    return subgraph(graph, keep)

def prepare_graph(graph):

    for k, n in graph._node.items():
    
        n['distance'] = 0
        n['time'] = 0
        n['price'] = 0

        # if 'station' not in str(k):

        #     n['min_soc'] = .5

    for s, a in graph._adj.items():
        keep = {}
        for t, l in a.items():

            l['price'] = 0
                
            if 'station' not in s:
                if l['distance'] >= 100e3:
                        keep[t] = l

            elif 'station' not in t:
                if l['distance'] >= 100e3:
                        keep[t] = l

            else:
                if 'station' in t:
                    if l['distance'] >= 200e3:
                        keep[t] = l
        
        graph._adj[s] = keep

    return graph

def make_tree(graph, origin):

    node_origin = graph._node[origin]

    for source in graph._node.keys():

        keep = {}

        for target, link in graph._adj[source].items():

            node_source = graph._node[source]
            node_target = graph._node[target]

            dist_source = (
                (node_origin['x'] - node_source['x']) ** 2 +
                (node_origin['y'] - node_source['y']) ** 2
                )

            dist_target = (
                (node_origin['x'] - node_target['x']) ** 2 +
                (node_origin['y'] - node_target['y']) ** 2
                )

            if dist_target >= dist_source * .95:

                keep[target] = link
        
        graph._adj[source] = keep

    return graph

def add_stations(graph, station, rng):

    for node in graph._node.values():

        if 'n_dcfc' in node.keys():

            station.recompute(chargers = node['n_dcfc'], rng = rng)

            # print(node['n_dcfc'], station.__dict__)

            # break

            node['update'] = station.update

    return graph

def accessibility(p_hat, field):

    accessibility = 0

    for key_0, value_0 in p_hat.items():

        for key_1, value_1 in value_0.items():

            accessibility += value_1[field].mean()

    accessibility /= len(p_hat) ** 2

    return accessibility