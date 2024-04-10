import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import pickle as pkl
import networkx as nx
import matplotlib.pyplot as plt

import src

def make_objects(graph):

    sg = src.experiment.tesla_network(graph)

    sg = src.experiment.prepare_graph(sg)

    return sg

print('Loading')

seed = 125897

rng = np.random.default_rng(seed = seed)

levels = {
    'rm': [1, 1.25, 1.5],
    'cr': [.75, .85, .95],
    'ra': [(0, .5), (0, 1), (.5, 1)],
}

design = src.utilities.full_factorial([len(v) for v in levels.values()]).astype(int)

atlas, graph, cities = src.experiment.load_graphs()

locations = list(cities.nodes)

sg = make_objects(graph)

print('Experiment')

for idx in range(len(design)):

    print(f'\nCase {idx} of {len(design)}\n')

    case = ([
        levels['rm'][design[idx][0]],
        levels['cr'][design[idx][1]],
        levels['ra'][design[idx][2]]
        ])

    print(f'\n {case} \n')

    vehicle = src.routing.ConstrainedVehicle(
        n_cases = 100,
        risk_attitude = case[2],
        capacity = 80 * 3.6e6 * .8 * case[0],
        efficiency = 536.4,
        rate = 170e3 * case[0],
    )

    station = src.routing.Station(
        vehicle,
        reliability = case[1],
    )

    sg = src.experiment.add_stations(sg, station, rng)

    expectations, values, paths = vehicle.all_pairs(
        sg,
        nodes = locations,
    )

    print('\n')
    print({k: v['time'].mean() / 3600 for k, v in values['Fresno'].items() if 'station' not in k})
    print('\n')

    pkl.dump(
        [case, expectations, values, paths],
        open(f'Outputs/Exp/Model_3_{idx}.pkl', 'wb')
    )