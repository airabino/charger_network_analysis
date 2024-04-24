'''
Module for Dijkstra routing

Code is based on (is an edited version of):
NetworkX shortest_paths.weighted._dijkstra_multisource

Edits are to allow for native tracking of multiple shortest path simultaneously.
For example, one could get a shortest path weighted by 'distance' but also
want to know path 'time', this edited code allows for this to be done efficiently.
'''
import time
import numpy as np
import networkx as nx

from copy import deepcopy
from heapq import heappop, heappush
from itertools import count
from sys import float_info, maxsize

from scipy.stats import t as t_dist
from scipy.stats import uniform, norm
from scipy.special import factorial

from .progress_bar import ProgressBar
from .rng import Queuing_Time_Distribution

from .dijkstra import dijkstra
from .bellman import bellman

def in_range(x, lower, upper):

    return (x >= lower) & (x <= upper)

class Vehicle():

    def __init__(self, **kwargs):

        self.capacity = kwargs.get('capacity', 80 * 3.6e6) # [J]
        self.efficiency = kwargs.get('efficiency', 550) # [J/m]
        self.soc_bounds = kwargs.get('soc_bounds', (0, 1)) # ([-], [-])
        self.initial_values = kwargs.get(
            'initial_values',
            {
                'time': 0, # [s]
                'distance': 0, # [m]
                'price': 0, # [$]
                'soc': 1, # [dim]
            },
        )

        self.range = (
            (self.soc_bounds[1] - self.soc_bounds[0]) * self.capacity / self.efficiency
            )

    def initial(self):

        return self.initial_values

    def infinity(self):

        return {k: np.inf for k in self.initial_values.keys()}

    def update(self, values, link, node):

        if (link['type'] == 'inter_station') and (link['distance'] < .75 * self.range):

            return None, False

        updated_values = values.copy()

        traversal_energy = self.efficiency * link['distance']
        traversal_delta_soc = traversal_energy / self.capacity

        updated_values['time'] += link['time']
        updated_values['distance'] += link['distance']
        updated_values['price'] += link['price']
        updated_values['soc'] -= traversal_delta_soc

        feasible = in_range(updated_values['soc'], *self.soc_bounds)

        return updated_values, feasible

    def compare(self, values, comparison):

        return values['time'], values['time'] < comparison['time']



def super_quantile_integral(x, p = (0, 1), n = 100):
    
    q = np.linspace(p[0], p[1], n)
    
    sq = 1/(p[1] - p[0]) * (np.quantile(x, q) * (q[1] - q[0])).sum()

    return sq

def super_quantile(x, p = (0, 1), n = 100):
    
    p_k = np.linspace(p[0], p[1], n)

    q_k = np.quantile(x, p_k)

    return q_k.mean(), q_k.std()

def super_quantile_fast(x, risk_attitude, n = 100):
    
    # q = np.linspace(risk_attitude[0], risk_attitude[1], n)

    # # print(q)

    # # print(norm(*norm.fit(x)).ppf(q))
    
    # sq = 1/(risk_attitude[1] - risk_attitude[0]) * (
    #     norm(*norm.fit(x)).ppf(q) * (q[1] - q[0])
    #     ).sum()

    sq = x.mean()

    return sq

def improvement(x, y, alpha):

    x_n = len(x)
    y_n = len(y)

    x_mu = x.mean()
    y_mu = y.mean()

    if x_mu >= y_mu:
        # print('a')

        return False

    x_sigma = x.std()
    y_sigma = y.std()

    x_se = x_sigma / np.sqrt(x_n)
    y_se = y_sigma / np.sqrt(y_n)

    x_y_se = np.sqrt(x_se ** 2 + y_se ** 2)

    t = (x_mu - y_mu) / x_y_se

    df = x_n + y_n

    p = (1 - t_dist.cdf(np.abs(t), df))*2

    print(t, df, p <= alpha)

    return p <= alpha



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

class Station():

    def __init__(self, vehicle, **kwargs):

        self.vehicle = vehicle

        self.seed = kwargs.get('seed', None)
        self.rng = kwargs.get('rng', np.random.default_rng())
        self.chargers = kwargs.get('chargers', 1)

        self.arrival_param = kwargs.get('arrival_param', (1, .5))
        self.arrival_limits = kwargs.get('arrival_limits', (.1, 1.9))

        self.service_param = kwargs.get('service_param', (60, 15))
        self.service_limits = kwargs.get('service_limits', (10, 110))
        
        self.reliability = kwargs.get('reliability', 1)
        self.energy_price = kwargs.get('energy_price', .5 / 3.6e6)

        self.populate()

    def recompute(self, **kwargs):

        for key, val in kwargs.items():

            setattr(self, key, val)

        self.populate()

    def clip_norm(self, param, limits):

        return np.clip(norm(*param).rvs(self.vehicle.n_cases), *limits)

    def populate(self):

        self.functional_chargers = self.functionality()
        self.at_least_one_functional_charger = self.functional_chargers > 0

        self.arrival = (
            3600 / (self.chargers * self.clip_norm(self.arrival_param, self.arrival_limits))
            )

        self.service = (
            self.clip_norm(self.service_param, self.service_limits) *
            3.6e6 / self.vehicle.rate
            )

        self.delay_time = np.zeros(self.vehicle.n_cases)

        for idx in range(self.vehicle.n_cases):

            self.delay_time[idx] = self.queuing_time(
                1 / self.arrival[idx],
                1 / self.service[idx],
                self.functional_chargers[idx]
                )

        self.functions = {
            'time': lambda x: self.time(x),
            'price': lambda x: self.price(x),
            'soc': lambda x: self.soc(x),
        }

    def update(self, x):

        x = self.time(x)
        x = self.price(x)
        x = self.soc(x)

        return x

    def queuing_time(self, l, m, c):

        rho = l / (c * m)

        k = np.arange(0, c, 1)

        p_0 = 1 / (
            sum([(c * rho) ** k / factorial(k) for k in k]) +
            (c * rho) ** c / (factorial(c) * (1 - rho))
        )

        l_q = (p_0 * (l / m) ** c * rho) / (factorial(c) * (1 - rho))

        w_q = l_q / l

        return np.nanmax([w_q, 0])

    def functionality(self):

        rn = self.rng.random(size = (self.chargers, self.vehicle.n_cases))

        return (rn <= self.reliability).sum(axis = 0)

    def time(self, x):
        
        time = (
            (self.vehicle.capacity * (1 - x['soc']) / self.vehicle.rate + self.delay_time)
            )

        # time = 0

        x['time'] += time

        return x

    def price(self, x):

        price = (
            (self.vehicle.capacity * (1 - x['soc']) * self.energy_price)
            )

        x['price'] += price

        return x

    def soc(self, x):

        x['soc'] += (1 - x['soc']) * self.at_least_one_functional_charger

        return x

# class Vehicle():

#     def __init__(self, **kwargs):

#         self.n_cases = kwargs.get('n_cases', 1) # [-]
#         self.risk_attitude = kwargs.get('risk_attitude', (0, 1)) # [-]
#         self.cutoff = kwargs.get('cutoff', np.inf) # [m]

#         if self.n_cases == 1:

#             self.populate_deterministic()

#         else:

#             self.populate_stochastic()

#     def all_pairs(self, graph, nodes = [], **kwargs):

#         expectations = {}
#         values = {}
#         paths = {}


#         if not nodes:

#             nodes = list(graph.nodes)

#         for node in ProgressBar(nodes, **kwargs.get('progress_bar', {})):

#             if kwargs.get('tree', False):

#                 sg = make_tree(graph, node)

#             else:

#                 sg = graph

#             expectations_n, values_n, paths_n = self.routes(
#                 sg, [node], destinations = nodes, **kwargs,
#                 )

#             expectations[node] = (
#                 {key: val for key, val in expectations_n.items() if key in nodes}
#                 )

#             values[node] = (
#                 {key: val for key, val in values_n.items() if key in nodes}
#                 )

#             if kwargs.get('return_paths', False):

#                 paths[node] = (
#                     {key: val for key, val in paths_n.items() if key in nodes}
#                     )

#         return expectations, values, paths

#     def routes(self, graph, origins, destinations = [], return_paths = False):

#         expectations, values, paths = dijkstra(
#             graph,
#             origins,
#             destinations = destinations,
#             states = self.states,
#             constraints = self.constraints,
#             objectives = self.objectives,
#             return_paths = return_paths,
#             )

#         return expectations, values, paths

#     def state_update(self, x, v):

#         x['time'] += v['time']
#         x['distance'] += v['distance']

#         return x

#     def populate_deterministic(self):

#         self.objectives = lambda x: x['time'],

#         self.constraints = lambda x: x['distance'] <= self.cutoff

#         self.states = {
#             'initial': {
#                 'time': 0,
#                 'distance': 0,
#             },
#             'update': lambda x, v: self.state_update(x, v),
#         }

#     def populate_stochastic(self):

#         self.objectives = lambda x: super_quantile(x['time'], self.risk_attitude),

#         self.constraints = lambda x: (
#                 super_quantile(x['distance'], self.risk_attitude) <= self.cutoff
#             )

#         self.states = {
#             'initial': {
#                 'time': np.array([0.] * self.n_cases),
#                 'distance': np.array([0.] * self.n_cases),
#             },
#             'update': lambda x, v: self.state_update(x, v),
#         }

class ConstrainedVehicle(Vehicle):

    def __init__(self, **kwargs):

        self.n_cases = kwargs.get('n_cases', 30) # [-]
        self.capacity = kwargs.get('capacity', 80 * 3.6e6) # [J]
        self.efficiency = kwargs.get('efficiency', 500) # [J/m]
        self.rate = kwargs.get('rate', 80e3) # [W]
        self.initial_soc = kwargs.get('initial_soc', 1.) # [-]
        self.max_soc = kwargs.get('max_soc', 1.) # [-]
        self.min_soc = kwargs.get('min_soc', .2) # [-]
        self.risk_attitude = kwargs.get('risk_attitude', (0, 1)) # [-]
        self.out_of_charge_penalty = kwargs.get('out_of_charge_penalty', 3*3600) # [s]

        if self.n_cases > 1:

            self.expectation_function = super_quantile

        else:

            self.expectation_function = lambda x, a: x[0]

        self.populate()

    def routes(self, graph, origins, destinations = [], return_paths = False, **kwargs):

        expectations, values, paths = dijkstra(
            graph,
            origins,
            destinations = destinations,
            states = self.states,
            constraints = self.constraints,
            objectives = self.objectives,
            return_paths = return_paths,
            )

        return expectations, values, paths

    def state_update(self, x, v):

        x['soc'] -= v['distance'] * self.efficiency /  self.capacity
        x['time'] += v['time']
        x['time_nc'] += v['time']
        x['distance'] += v['distance']
        x['price'] += v['price']

        return x

    def constraint_test(self, x, v):

        feasible = in_range(
            self.expectation_function(x['soc'],
                (1 - self.risk_attitude[0], 1 - self.risk_attitude[1])),
            v.get('min_soc', self.min_soc),
            v.get('max_soc', self.max_soc)
        )

        return True, x

    def populate(self):

        self.objectives = lambda x: self.expectation_function(x['time'], self.risk_attitude)

        self.constraints = lambda x, v: self.constraint_test(x, v)

        self.states = {
            'initial': {
                'soc': np.array([self.initial_soc] * self.n_cases),
                'time': np.array([0.] * self.n_cases),
                'time_nc': np.array([0.] * self.n_cases),
                'distance': np.array([0.] * self.n_cases),
                'price': np.array([0.] * self.n_cases),
            },
            'update': lambda x, v: self.state_update(x, v),
        }