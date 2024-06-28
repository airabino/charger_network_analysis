import numpy as np

from .routing import Vehicle, Station, all_pairs_shortest_paths, supply_costs
from .utilities import full_factorial

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
        'cases': 100,
        'type': 'ac',
        'access': 'private',
        'price': .4 / 3.6e6,
        'setup_time': 0,
        'rng': lambda rng: rng,
    },
    'station': {
        'reliability': lambda rng: rng.random() * .5 + .5,
        'cases': 100,
        'type': 'dc',
        'access': 'public',
        'power': _network_power,
        'price': .5 / 3.6e6,
        'setup_time': 300,
        'rng': lambda rng: rng,
    },
}

def generate_case(graphs, vehicle_param, station_param, rng = np.random.default_rng()):

    graph_index = rng.choice(list(range(len(graphs))))

    vehicle_kw = {}
    station_kw = {}

    for key, val in vehicle_param.items():

        if callable(val):

            vehicle_kw[key] = val(rng)

        else:

            vehicle_kw[key] = val

    for key, val in station_param.items():

        station_kw[key] = {}

        for k, v in val.items():

            if callable(v):

                station_kw[key][k] = v(rng)

            else:

                station_kw[key][k] = v

    return graph_index, vehicle_kw, station_kw


def run_case(graph, vehicle_kw, station_kw, method = 'dijkstra'):

    vehicle = Vehicle(**vehicle_kw)

    origins = [k for k, v in graph._node.items() if v['type'] == 'place']

    graph = supply_costs(graph, vehicle, station_kw)
    
    costs, values, paths = all_pairs_shortest_paths(
        graph, origins,
        objective = vehicle,
        method = method,
        progress_bar_kw = {'disp': False},
    )

    return costs, values, paths