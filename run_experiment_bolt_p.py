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

    sg = src.experiment.non_proprietary_network(graph)

    sg = src.experiment.prepare_graph(sg)

    return sg

idx = int(sys.argv[1])

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

case = ([
    levels['rm'][design[idx][0]],
    levels['cr'][design[idx][1]],
    levels['ra'][design[idx][2]]
    ])

vehicle = src.routing.ConstrainedVehicle(
    n_cases = 100,
    risk_attitude = case[2],
    capacity = 65 * 3.6e6 * .8 * case[0],
    efficiency = 626.5,
    rate = 55e3 * case[0],
)

station = src.routing.Station(
    vehicle,
    reliability = case[1],
)

sg = src.experiment.add_stations(sg, station, rng)

print(f'running {idx}')

expectations, values, paths = vehicle.all_pairs(
    sg,
    nodes = locations,
    progress_bar = {
        'disp': False,
    }
)

pkl.dump(
    [case, expectations, values, paths],
    open(f'Outputs/Exp/Bolt_{idx}.pkl', 'wb')
)

print(f'finished {idx}')