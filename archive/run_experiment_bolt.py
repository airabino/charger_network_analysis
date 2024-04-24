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

print('Loading')

atlas, graph, cities = src.experiment.load_graphs()

locations = list(cities.nodes)

sg_c = src.experiment.prepare_graph(graph)

sg_np = src.experiment.non_proprietary_network(graph)

sg_np = src.experiment.prepare_graph(sg_np)

sg_t = src.experiment.non_proprietary_network(graph)

sg_t = src.experiment.prepare_graph(sg_t)



seed = 125897

rng = np.random.default_rng(seed = seed)


model_3 = src.routing.ConstrainedVehicle(
    n_cases = 100,
    risk_attitude = (0, 1),
    capacity = 80 * 3.6e6 * .8,
    efficiency = 536.4,
    rate = 170e3,
)

station_model_3 = src.routing.Station(
    model_3,
    reliability = .85,
)

bolt = src.routing.ConstrainedVehicle(
    n_cases = 100,
    risk_attitude = (0, 1),
    capacity = 65 * 3.6e6 * .8,
    efficiency = 626.5,
    rate = 55e3,
)

station_bolt = src.routing.Station(
    bolt,
    reliability = .85,
)


print('Experiment')

print('Bolt np')

sg = src.experiment.add_stations(sg_np, station_bolt, rng)

expectations, values, paths = bolt.all_pairs(
    sg,
    nodes = locations,
)

print('\n')
print({k: v['time'].mean() / 3600 for k, v in values['Fresno'].items() if 'station' not in k})
print('\n')

pkl.dump(
    [case, expectations, values, paths],
    open('ExpResults/Bolt_np.pkl', 'wb')
)

print('Bolt t')

sg = src.experiment.add_stations(sg_t, station_bolt, rng)

expectations, values, paths = bolt.all_pairs(
    sg,
    nodes = locations,
)

print('\n')
print({k: v['time'].mean() / 3600 for k, v in values['Fresno'].items() if 'station' not in k})
print('\n')

pkl.dump(
    [case, expectations, values, paths],
    open('ExpResults/Bolt_t.pkl', 'wb')
)

print('Bolt c')

sg = src.experiment.add_stations(sg_c, station_bolt, rng)

expectations, values, paths = bolt.all_pairs(
    sg,
    nodes = locations,
)

print('\n')
print({k: v['time'].mean() / 3600 for k, v in values['Fresno'].items() if 'station' not in k})
print('\n')

pkl.dump(
    [case, expectations, values, paths],
    open('ExpResults/Bolt_c.pkl', 'wb')
)


print('Model 3 np')

sg = src.experiment.add_stations(sg_np, station_model_3, rng)

expectations, values, paths = model_3.all_pairs(
    sg,
    nodes = locations,
)

print('\n')
print({k: v['time'].mean() / 3600 for k, v in values['Fresno'].items() if 'station' not in k})
print('\n')

pkl.dump(
    [case, expectations, values, paths],
    open('ExpResults/Model_3_np.pkl', 'wb')
)

print('Model 3 t')

sg = src.experiment.add_stations(sg_t, station_model_3, rng)

expectations, values, paths = model_3.all_pairs(
    sg,
    nodes = locations,
)

print('\n')
print({k: v['time'].mean() / 3600 for k, v in values['Fresno'].items() if 'station' not in k})
print('\n')

pkl.dump(
    [case, expectations, values, paths],
    open('ExpResults/Model_3_t.pkl', 'wb')
)

print('Model 3 c')

sg = src.experiment.add_stations(sg_c, station_model_3, rng)

expectations, values, paths = model_3.all_pairs(
    sg,
    nodes = locations,
)

print('\n')
print({k: v['time'].mean() / 3600 for k, v in values['Fresno'].items() if 'station' not in k})
print('\n')

pkl.dump(
    [case, expectations, values, paths],
    open('ExpResults/Model_3_c.pkl', 'wb')
)