import time
import numpy as np

from copy import deepcopy
from scipy.stats import norm
from scipy.special import factorial

from .progress_bar import ProgressBar
from .dijkstra import dijkstra
from .bellman import bellman

def in_range(x, lower, upper):

    return (x >= lower) & (x <= upper)

def super_quantile(x, p = (0, 1), n = 100):
    
    p_k = np.linspace(p[0], p[1], n)

    q_k = np.quantile(x, p_k)

    return np.nan_to_num(q_k.mean(), nan = np.inf)

def shortest_paths(graph, origins, method = 'dijkstra', **kwargs):
    '''
    Return path costs, path values, and paths using Dijkstra's or Bellman's method

    Produces paths to each destination from closest origin

    Depends on an Objective object which contains the following four functions:

    values = initial() - Function which produces the starting values of each problem state
    to be applied to the origin node(s)

    values = infinity() - Function which produces the starting values for each non-origin
    node. The values should be intialized such that they are at least higher than any
    conceivable value which could be attained during routing.

    values, feasible = update(values, edge, node) - Function which takes current path state
    values and updates them based on the edge traversed and the target node and whether the
    proposed edge traversal is feasible. This function returns the values argument and a
    boolean feasible.

    values, savings = compare(values, approximation) - Function for comparing path state
    values with the existing best approximation at the target node. This function returns
    the values argument and a boolean savings.
    '''

    if method == 'dijkstra':

        costs, values, paths = dijkstra(graph, origins, **kwargs)

    elif method == 'bellman':

        costs, values, paths = bellman(graph, origins, **kwargs)

    destinations = kwargs.get('destinations', [])
    
    if destinations:

        costs_d = {}
        values_d = {}
        paths_d = {}

        for destination in destinations:

            costs_d[destination] = costs[destination]
            values_d[destination] = values[destination]
            paths_d[destination] = paths[destination]

        return costs_d, values_d, paths_d

    else:

        return costs, values, paths

def all_pairs_shortest_paths(graph, origins, method = 'dijkstra', **kwargs):
    '''
    Return path costs, path values, and paths using Dijkstra's or Bellman's method

    Produces paths to each origin from each origin

    Depends on an Objective object which contains the following four functions:

    values = initial() - Function which produces the starting values of each problem state
    to be applied to the origin node(s)

    values = infinity() - Function which produces the starting values for each non-origin
    node. The values should be intialized such that they are at least higher than any
    conceivable value which could be attained during routing.

    values, feasible = update(values, edge, node) - Function which takes current path state
    values and updates them based on the edge traversed and the target node and whether the
    proposed edge traversal is feasible. This function returns the values argument and a
    boolean feasible.

    values, savings = compare(values, approximation) - Function for comparing path state
    values with the existing best approximation at the target node. This function returns
    the values argument and a boolean savings.
    '''

    # print(origins)
    # print(kwargs.get('progress_bar_kw', {}))

    if method == 'dijkstra':

        routing_function = dijkstra

    elif method == 'bellman':

        routing_function = bellman

    costs = {}
    values = {}
    paths = {}

    for origin in ProgressBar(origins, **kwargs.get('progress_bar_kw', {})):

        result = routing_function(graph, [origin], destinations = origins, **kwargs)

        costs[origin] = result[0]
        values[origin] = result[1]
        paths[origin] = result[2]

    return costs, values, paths

def specific_road_trip_accessibility(values, field = 'time', expectation = np.mean):

    sum_cost = 0

    n = len(values)

    for key, val in values.items():

        sum_cost += expectation(np.atleast_1d(val[field]))

    return sum_cost / n



class Objective():

    def __init__(self, field = 'weight', limit = np.inf):

        self.field = field
        self.limit = limit

    def initial(self):

        return 0

    def infinity(self):

        return np.inf

    def update(self, values, link, node):

        values += link.get(self.field, 1)

        return values, values <= self.limit

    def compare(self, values, approximation):

        return values, values < approximation

class Vehicle():

    def __init__(self, **kwargs):

        self.capacity = kwargs.get('capacity', 80 * 3.6e6) # [J]
        self.efficiency = kwargs.get('efficiency', 550) # [J/m]
        self.charge_rate = kwargs.get('charge_rate', 80e3) # [W]
        self.charge_time_penalty = kwargs.get('charge_time_penalty', 300) # [s]
        self.soc_bounds = kwargs.get('soc_bounds', (0, 1)) # ([-], [-])
        self.max_charge_start_soc = kwargs.get('max_charge_start_soc', 1) # [-]

        self.initial_values = kwargs.get(
            'initial_values',
            {
                'time': 0, # [s]
                'driving_time': 0, # [s]
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

        if (
            (node['type'] == 'station') and
            not in_range(
                link['distance'],
                (1 - self.max_charge_start_soc) * self.range,
                self.range
            )):

            return values, False

        updated_values = values.copy()

        traversal_energy = self.efficiency * link['distance']
        traversal_delta_soc = traversal_energy / self.capacity

        updated_values['time'] += link['time']
        updated_values['driving_time'] += link['time']
        updated_values['distance'] += link['distance']
        updated_values['price'] += link['price']
        updated_values['soc'] -= traversal_delta_soc

        feasible = in_range(updated_values['soc'], *self.soc_bounds)

        if node['type'] == 'station':

            updated_values = node['station'].update(updated_values)

            # charge_rate = max([self.charge_rate, node.get('charge_rate', 0)])

            # delay = node.get('expected_delay', 0)

            # delta_soc = self.soc_bounds[1] - updated_values['soc']

            # updated_values['soc'] += delta_soc
            # updated_values['time'] += (
            #     delta_soc * self.capacity / charge_rate + self.charge_time_penalty + delay
            #     )
            # updated_values['price'] += (
            #     delta_soc * self.capacity * node.get('charge_price', 0)
            #     )

        return updated_values, feasible

    def compare(self, values, approximation):

        return values['time'], values['time'] < approximation['time']

class StochasticVehicle():

    def __init__(self, **kwargs):

        self.cases = kwargs.get('cases', 1) # [-]
        self.capacity = kwargs.get('capacity', 80 * 3.6e6) # [J]
        self.efficiency = kwargs.get('efficiency', 550) # [J/m]
        self.charge_rate = kwargs.get('charge_rate', 80e3) # [W]
        self.soc_bounds = kwargs.get('soc_bounds', (0, 1)) # ([-], [-])
        self.charge_target_soc = kwargs.get('charge_target_soc', 1) # [-]
        self.max_charge_start_soc = kwargs.get('max_charge_start_soc', 1) # [-]
        self.risk_attitude = kwargs.get('risk_attitude', (0, 1)) # ([-], [-])

        if self.cases == 1:

            self.expectation = kwargs.get(
                'expectation',
                lambda x: x[0],
                )

        else:

            self.expectation = kwargs.get(
                'expectation',
                lambda x: super_quantile(x, self.risk_attitude),
                )
            
        self.initial_values = kwargs.get(
            'initial_values',
            {
                'time': np.zeros(self.cases), # [s]
                'driving_time': np.zeros(self.cases), # [s]
                'distance': np.zeros(self.cases), # [m]
                'price': np.zeros(self.cases), # [$]
                'soc': np.ones(self.cases), # [dim]
            },
        )

        self.range = (
            (self.soc_bounds[1] - self.soc_bounds[0]) * self.capacity / self.efficiency
            )

    def select_case(self, case):

        new_object = deepcopy(self)
        new_object.expectation = lambda x: x[case]

        return new_object

    def initial(self):

        return self.initial_values

    def infinity(self):

        return {k: np.ones(self.cases) * np.inf for k in self.initial_values.keys()}

    def update(self, values, link, node):

        if (
            (node['type'] == 'station') and
            not in_range(
                link['distance'],
                (1 - self.max_charge_start_soc) * self.range,
                self.range
            )):

            return values, False

        traversal_energy = self.efficiency * link['distance']
        traversal_delta_soc = traversal_energy / self.capacity

        updated_values = {}

        updated_values['time'] = values['time'] + link['time']
        updated_values['driving_time'] = values['driving_time'] + link['time']
        updated_values['distance'] = values['distance'] + link['distance']
        updated_values['price'] = values['price'] + link['price']
        updated_values['soc']  = values['soc'] - traversal_delta_soc

        # print(updated_values['soc'][0])
        # print(self.expectation(updated_values['soc']))

        feasible = in_range(self.expectation(updated_values['soc']), *self.soc_bounds)

        if node['type'] == 'station':

            updated_values = node['station'].update(
                updated_values, self,
                )

        return updated_values, feasible

    def compare(self, values, comparison):

        return (
            self.expectation(values['time']), 
            self.expectation(values['time']) < self.expectation(comparison['time'])
            )

class StochasticStation():

    def __init__(self, **kwargs):

        self.cases = kwargs.get('cases', 1) # [-]
        self.seed = kwargs.get('seed', None)
        self.rng = kwargs.get('rng', np.random.default_rng(self.seed))
        self.chargers = kwargs.get('chargers', 1)
        self.charge_rate = kwargs.get('charge_rate', 80e3)
        self.charge_price = kwargs.get('charge_price', .5 / 3.6e6) # [$/J]
        self.base_delay = kwargs.get('base_delay', 0) # [s]

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

        return np.clip(self.rng.normal(*param, size = self.cases), *limits)

    def populate(self):

        self.functional_chargers = self.functionality()
        self.at_least_one_functional_charger = self.functional_chargers > 0

        self.arrival = (
            3600 / (self.chargers * self.clip_norm(self.arrival_param, self.arrival_limits))
            )

        self.service = (
            self.clip_norm(self.service_param, self.service_limits) *
            3.6e6 / self.charge_rate
            )

        self.delay_time = np.zeros(self.cases) + self.base_delay

        for idx in range(self.cases):

            self.delay_time[idx] += self.queuing_time(
                1 / self.arrival[idx],
                1 / self.service[idx],
                self.functional_chargers[idx]
                )

        self.functions = {
            'time': lambda x: self.time(x),
            'price': lambda x: self.price(x),
            'soc': lambda x: self.soc(x),
        }

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

        rn = self.rng.random(size = (self.chargers, self.cases))

        return (rn <= self.reliability).sum(axis = 0)

    def update(self, x, vehicle):

        capacity = vehicle.capacity
        target = vehicle.charge_target_soc
        rate = min([self.charge_rate, vehicle.charge_rate])

        x = self.time(x, capacity, rate, target)
        x = self.price(x, capacity, rate, target)
        x = self.soc(x, capacity, rate, target)

        return x

    def time(self, x, capacity, rate, target):
        
        time = (
            (capacity * (target - x['soc']) /
                rate +
                self.delay_time)
            )

        x['time'] += time

        return x

    def price(self, x, capacity, rate, target):

        price = (
            (capacity * (target - x['soc']) * self.charge_price)
            )

        x['price'] += price

        return x

    def soc(self, x, capacity, rate, target):

        x['soc'] += (target - x['soc']) * self.at_least_one_functional_charger

        return x