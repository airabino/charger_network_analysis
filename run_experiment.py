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

def model_3(graph):

    vehicle = src.routing.ConstrainedVehicle(
        n_cases = 100,
        risk_attitude = (0, 1),
        ess_capacity = 80 * 3.6e6 * .8,
        efficiency = 536.4,
        rate = 170e3,
    )

    station = src.routing.Station(
        vehicle,
        reliability = .75,
        rate = 170e3,
        seed = 125897,
    )

    sg = src.experiment.tesla_network(graph)

    sg = src.experiment.prepare_graph(sg)

    sg = src.experiment.add_stations(sg, station)

    return vehicle, station, sg

print('Loading')

levels = {
    'rm': [1, 1.25, 1.5],
    'cr': [.75, .85, .95],
    'ra': [(0, .5), (0, 1), (.5, 1)],
}

design = src.utilities.full_factorial([len(v) for v in levels.values()]).astype(int)

atlas, graph, cities = src.experiment.load_graphs()

locations = list(cities.nodes)

vehicle, station, sg = model_3(graph)

print('Experiment')

for idx in range(len(design)):

    print(f'Case {idx} of {len(design)}')

    case = ([
        levels['rm'][levels[idx][0]],
        levels['cr'][levels[idx][1]],
        levels['ra'][levels[idx][2]]
        ])

    expectations, values, paths = veh.all_pairs(
        sg,
        nodes = locations,
    )

    pkl.dump(
        [case, expectations, values, paths],
        open(f'Outputs/Exp/Model_3_{idx}.pkl', 'wb')
    )