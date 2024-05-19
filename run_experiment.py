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
    '-i', '--indices',
    default = [],
    nargs = '+',
    )


def main(indices = []):

    #Loading SNGs

    non_proprietary_sng_us = src.graph.graph_from_json(
        'Outputs/SNG/non_proprietary_sng_us.json'
        )

    tesla_sng_us = src.graph.graph_from_json(
        'Outputs/SNG/tesla_sng_us.json'
        )

    combined_sng_us = src.graph.graph_from_json(
        'Outputs/SNG/combined_sng_us.json'
        )

    # print(tesla_sng_us.nodes)

    # Making cases

    risk_attitudes = [(0, .5), (0, 1), (.5, 1)]
    capacities = [49.16 * 3.6e6, 73.75 * 3.6e6, 98.33 * 3.6e6]
    charge_rates = [c / 3.6e3 / .75 for c in capacities]
    reliabilities = [.50, .75, 1]
    sngs = [tesla_sng_us, non_proprietary_sng_us, combined_sng_us]

    levels = src.utilities.full_factorial([3, 3, 3, 3])

    vehicle_kw = []
    station_kw = []

    for case in levels:

        vehicle_kw.append({
            'capacity': capacities[case[1]],
            'risk_attitude': risk_attitudes[case[0]],
            'efficiency': 550,
            'charge_rate': charge_rates[case[1]],
            'cases': 1,
        })
        
        station_kw.append({
            'cases': 100,
            'charge_rate': charge_rates[case[1]],
            'reliability': reliabilities[case[2]],
            'base_delay': 60,
        })

    if not indices:

        indices = range(len(cases))

    for idx in indices:

        costs, values, paths = src.experiments.run_case(
            sngs[levels[idx][3]], vehicle_kw[idx], station_kw[idx], method = 'bellman'
        )

        pkl.dump([costs, values, paths], open(f'Outputs/Experiment/case_{idx}.pkl', 'wb'))

if __name__ == '__main__':

    args = vars(parser.parse_args(sys.argv[1:]))

    indices = [eval(idx) for idx in args["indices"]]

    if len(indices) == 2:

        indices = list(range(*indices))

    # print(indices)

    t0 = time.time()

    print(f'Running cases: {indices}')

    main(indices = indices)

    print(f'Executed cases {indices} in {time.time() - t0:.4f} seconds')



