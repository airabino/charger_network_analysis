import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import pickle as pkl
import networkx as nx
import matplotlib.pyplot as plt

from pprint import pprint

import src
from src.reload import deep_reload

parser = argparse.ArgumentParser()

parser.add_argument(
    '-c', '--cases',
    default = 1,
    type = int,
    )

parser.add_argument(
    '-i', '--index',
    default = 0,
    type = int,
    )

parser.add_argument(
    '-s', '--seed',
    default = None,
    type = int,
    )

_vehicle_kwargs = {
    'capacity': lambda rng: (rng.random() * 80 + 40) * 3.6e6,
    'power': lambda rng: (rng.random() * 150 + 50) * 1e3,
    'risk_attitude': lambda rng: (rng.random() * .8 + .1) + np.array([-.1, .1]),
    'cases': 1,
    'soc_bounds': (.1, 1),
    'efficiency': 550,
    'linear_fraction': .8,
}

_network_power = {
    'Tesla': [250e3],
    'Electrify America': [150e3],
    'ChargePoint Network': [62.5e3],
    'eVgo Network': [50e3, 100e3, 350e3],
    'default': [50e3],
}

_station_kwargs = {
    'place': {
        'cases': 30,
        'type': 'ac',
        'access': 'private',
        'price': .4 / 3.6e6,
        'setup_time': 0,
        'rng': lambda rng: rng,
    },
    'station': {
        'reliability': lambda rng: rng.random() * .5 + .5,
        'cases': 30,
        'type': 'dc',
        'access': 'public',
        'power': _network_power,
        'price': .5 / 3.6e6,
        'setup_time': 300,
        'traffic': lambda rng: rng.random(),
        'rng': lambda rng: rng,
    },
}

sng_combined = src.graph.graph_from_json('Outputs/graph_full_red.json')
sng_tesla = src.graph.graph_from_json('Outputs/graph_full_tesla.json')
sng_other = src.graph.graph_from_json('Outputs/graph_full_non_tesla.json')

graphs = [sng_combined, sng_tesla, sng_other]

places = [k for k, v in sng_combined._node.items() if v['type'] == 'place']
nodes = list(sng_combined.nodes)

output_directory = '/media/aaron/Extreme SSD/random_experiment/'

def main(index = 0, cases = 1, seed = None):

    rng = np.random.default_rng(seed)

    for idx in range(index, index + cases):

        _, vehicle_kw, station_kw = src.experiments.generate_case(
            graphs, _vehicle_kwargs, _station_kwargs, rng = rng,
        )

        for idx_graph in range(len(graphs)):

            costs, values, paths = src.experiments.run_case(
                graphs[idx_graph], vehicle_kw, station_kw, method = 'dijkstra',
            )

            out_arrays = src.analysis.values_to_array(
                places, values, ['total_time', 'driving_time', 'charge_event']
            )

            visits = src.analysis.nodes_visited(
                nodes, paths,
            )

            pkl.dump(
                [idx_graph, vehicle_kw, station_kw, out_arrays, visits],
                open(output_directory + f'case_{idx}_{idx_graph}.pkl', 'wb')
                )

if __name__ == '__main__':

    args = vars(parser.parse_args(sys.argv[1:]))

    index = args['index']
    cases = args['cases']
    seed = args['seed']

    start = index
    finish = index + cases

    t0 = time.time()

    print(f"Running cases {start} through {finish}")

    main(index = index, cases = cases, seed = seed)

    print(
        f"Executed cases {start} through {finish}" + 
        f' in {(time.time() - t0) / 60:.4f} minutes'
        )



