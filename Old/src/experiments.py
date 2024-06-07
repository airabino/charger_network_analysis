import numpy as np

from .routing import StochasticVehicle, StochasticStation, all_pairs_shortest_paths
from .utilities import full_factorial

default_vehicle_param = {
    'capacity': lambda rng: (rng.random() * 80 + 40) * 3.6e6,
    'charge_rate': lambda rng: (rng.random() * 150 + 50) * 1e3,
    'risk_attitude': lambda rng: (rng.random() * .8 + .1) + np.array([-.1, .1]),
    'cases': lambda rng: 1,
    'charge_target_soc': lambda rng: .8,
    'soc_bounds': lambda rng: (.1, 1),
    'efficiency': lambda rng: 550,
}

default_station_param = {
    'reliability': lambda rng: rng.random() * .5 + .5,
    'base_delay': lambda rng: 60,
    'cases': lambda rng: 100,
    'arrival_param': lambda rng: (rng.random() * 2 + 1) * np.array([1, .25]),
    'service_param': lambda rng: (45 * 3.6e6, 0),
}

def generate_case(graphs, vehicle_param, station_param, rng = np.random.default_rng()):

    graph_index = rng.choice(list(range(len(graphs))))

    vehicle_kw = {}
    station_kw = {}

    for key, fun in vehicle_param.items():

        vehicle_kw[key] = fun(rng)

    for key, fun in station_param.items():

        station_kw[key] = fun(rng)

    return graph_index, vehicle_kw, station_kw


def run_case(graph, vehicle_kw, station_kw, seed = None, method = 'dijkstra'):

    vehicle = StochasticVehicle(**vehicle_kw)

    cities = [k for k, v in graph._node.items() if v['type'] == 'city']
    
    for source, node in graph._node.items():
    
        if node['type'] == 'station':
    
            node['station'] = StochasticStation(
                chargers = node['n_dcfc'], **station_kw, seed = seed,
            )

            if vehicle.cases == 1:

                node['station'].expect(vehicle)
    
    costs, values, paths = all_pairs_shortest_paths(
        graph, cities,
        objective = vehicle,
        method = method,
        return_paths = True,
        progress_bar_kw = {'disp': False},
    )

    return costs, values, paths