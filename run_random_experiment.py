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
    default = 123853692,
    type = int,
    )

vehicle_param = {
    'capacity': lambda rng: (rng.random() * 80 + 40) * 3.6e6,
    'charge_rate': lambda rng: (rng.random() * 150 + 50) * 1e3,
    'risk_attitude': lambda rng: np.sort(rng.random(size = (2, ))),
    'cases': lambda rng: 1,
    'charge_target_soc': lambda rng: .8,
    'soc_bounds': lambda rng: (.1, 1),
    'efficiency': lambda rng: 550,
}

station_param = {
    'reliability': lambda rng: rng.random(),
    'base_delay': lambda rng: 60,
    'cases': lambda rng: 100,
}

sng_combined = src.graph.graph_from_json('Outputs/sng_combined.json')
sng_tesla = src.graph.graph_from_json('Outputs/sng_tesla.json')
sng_other = src.graph.graph_from_json('Outputs/sng_other.json')

graphs = [sng_combined, sng_tesla, sng_other]

def main(index = 0, cases = 1, seed = None):

    rng = np.random.default_rng(seed)

    for idx in range(index, index + cases):

        graph_index, vehicle_kw, station_kw = src.experiments.generate_case(
            graphs, vehicle_param, station_param, rng = rng,
        )

        costs, values, paths = src.experiments.run_case(
            graphs[graph_index], vehicle_kw, station_kw,
        )

        pkl.dump(
            [graph_index, vehicle_kw, station_kw, costs, values, paths],
            open(f'Outputs/Random_Experiment/case_{idx}.pkl', 'wb')
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
        f' in {time.time() - t0:.4f} seconds'
        )



