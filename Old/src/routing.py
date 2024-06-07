import time
import numpy as np

from copy import deepcopy
from scipy.stats import norm
from scipy.special import factorial

from .progress_bar import ProgressBar
from .dijkstra import dijkstra, multi_directional_dijkstra
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

    # print(graph.nodes)

    if method == 'multi_dijkstra':

        return multi_directional_dijkstra(graph, origins, **kwargs)

    else:

        if method == 'dijkstra':

            routing_function = dijkstra

        elif method == 'bellman':

            routing_function = bellman

        costs = {}
        values = {}
        paths = {}

        for origin in ProgressBar(origins, **kwargs.get('progress_bar_kw', {})):

            result = shortest_paths(
                graph, [origin],
                destinations = origins,
                method = method, 
                **kwargs
                )

            costs[origin] = result[0]
            values[origin] = result[1]
            paths[origin] = result[2]

        return costs, values, paths

def impedance(values, origins = {}, destinations = {}, **kwargs):

    field = kwargs.get('field', 'time')
    expectation = kwargs.get('expectation', np.mean)
    constant = kwargs.get('constant', 1)


    if not origins:

        origins = {k: 1 for k in values.keys()}

    if not destinations:

        destinations = {k: 1 for k in values.keys()}

    sum_cost = 0

    n = 0

    for origin, mass_o in origins.items():

        for destination, mass_d in destinations.items():

            if origin != destination:

                sum_cost += (
                    constant * mass_o * mass_d *
                    expectation(np.atleast_1d(values[origin][destination][field]))
                    )

            n += 1

            # print(sum_cost)

    return sum_cost / n

def current(values, origins = {}, destinations = {}, **kwargs):

    field = kwargs.get('field', 'time')
    expectation = kwargs.get('expectation', np.mean)
    constant = kwargs.get('constant', 1)


    if not origins:

        origins = {k: 1 for k in values.keys()}

    if not destinations:

        destinations = {k: 1 for k in values.keys()}

    sum_cost = 0

    n = 0

    total_weight = sum([v for v in origins.values()])

    for origin, weight_o in origins.items():

        for destination, weight_d in destinations.items():

            if origin != destination:

                # print(constant * (voltage_d - voltage_o) )

                sum_cost += (
                    constant * weight_o * weight_d /
                    expectation(
                        np.atleast_1d(values[destination][origin][field]) * total_weight
                        )
                    )

            n += 1

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

def edge_types(graph):

    _adj = graph._adj
    _node = graph._node

    for source, adj in _adj.items():
        for target, edge in adj.items():

            edge['type'] = (
                f"{_node[source].get('type', 'none')}_{_node[target].get('type', 'none')}"
                )

    return graph

class Scout():

    def __init__(self, **kwargs):

        self.field = kwargs.get('field', 'time')
        self.limit = kwargs.get('limit', np.inf)
        self.edge_limit = kwargs.get('edge_limit', np.inf)
        self.exclude = kwargs.get('exclude', ['city_city'])
        self.disambiguation = kwargs.get('disambiguation', 0)

    def initial(self):

        return 0

    def infinity(self):

        return np.inf

    def update(self, values, edge, node):

        if (edge['type'] in self.exclude) or (edge[self.field] > self.edge_limit):

            return values, False

        values += edge.get(self.field, 1)

        return values, values <= self.limit

    def combine(self, values_0, values_1):

        return values_0 + values_1

    def compare(self, values, approximation):

        if approximation == np.inf:

            return values, values < approximation

        return values, values * (1 + self.disambiguation) < approximation

class Vehicle():

    def __init__(self, **kwargs):

        self.ddd = [0, 0]

        self.cases = kwargs.get('cases', 1) # [-]
        self.capacity = kwargs.get('capacity', 80 * 3.6e6) # [J]
        self.efficiency = kwargs.get('efficiency', 550) # [J/m]
        self.charge_rate = kwargs.get('charge_rate', 80e3) # [W]
        self.soc_bounds = kwargs.get('soc_bounds', (0, 1)) # ([-], [-])
        self.charge_target_soc = kwargs.get('charge_target_soc', 1) # [-]
        self.max_charge_start_soc = kwargs.get(
            'max_charge_start_soc', self.charge_target_soc) # [-]
        self.risk_attitude = kwargs.get('risk_attitude', (0, 1)) # ([-], [-])
        self.out_of_charge_penalty = kwargs.get('out_of_charge_penalty', 4 * 3600) # [s]

        self.field = kwargs.get('field', 'time')

        if self.cases == 1:

            self.expectation = kwargs.get(
                'expectation',
                lambda x: x[0],
                )

            self.expectation_reverse = kwargs.get(
                'expectation_reverse',
                lambda x: x[0],
                )

        else:

            self.expectation = kwargs.get(
                'expectation',
                lambda x: super_quantile(x, self.risk_attitude),
                )

            self.expectation_reverse = kwargs.get(
                'expectation_reverse',
                lambda x: super_quantile(
                    x,
                    (1 - self.risk_attitude[1], 1 - self.risk_attitude[0])),
                )
            
        self.initial_values = kwargs.get(
            'initial_values',
            {
                'time': np.zeros(self.cases), # [s]
                'merge_time': np.zeros(self.cases), # [s]
                'routing_time': np.zeros(self.cases), # [s]
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

    def combine(self, values_0, values_1):

        merged_values = {k: values_0[k] + values_1[k] for k in values_0.keys()}

        merged_values['soc'] = values_0['soc'] - (1 - values_1['soc'])
        # merged_values['time'] = values_0['merge_time'] + values_1['time']

        feasible = in_range(self.expectation_reverse(merged_values['soc']), *self.soc_bounds)

        return merged_values, feasible

    def initial(self):

        return self.initial_values

    def infinity(self):

        return {k: np.ones(self.cases) * np.inf for k in self.initial_values.keys()}

    def update(self, values, link, node):

        if (
            (node['type'] != 'city') and
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
        updated_values['merge_time'] = values['time'] + link['time']
        updated_values['routing_time'] = values['routing_time'] + link['time']
        updated_values['driving_time'] = values['driving_time'] + link['time']
        updated_values['distance'] = values['distance'] + link['distance']
        updated_values['price'] = values['price'] + link['price']
        updated_values['soc']  = values['soc'] - traversal_delta_soc

        feasible = in_range(
            self.expectation_reverse(updated_values['soc']),
            *self.soc_bounds
            )

        if not feasible:

            updated_values['time'] += self.out_of_charge_penalty
            updated_values['routing_time'] += self.out_of_charge_penalty
            # updated_values['driving_time'] += self.out_of_charge_penalty

            feasible = True

        if node['type'] == 'station':

            updated_values = node['station'].update(
                updated_values, self,
                )

        return updated_values, feasible

    def compare(self, values, comparison):

        values_exp = self.expectation(values[self.field])
        comparison_exp = self.expectation(comparison[self.field])

        savings = values_exp < comparison_exp

        return values_exp, savings

def mms_queue(self, arrival_rate, service_rate, servers):

    rho = arrival_rate / (service_rate * servers)

    probability_empty = 0

    for k in range(servers):

        probability_empty += (arrival_rate / service_rate) ** k / factorial(k)

    probability_empty += (
        (arrival_rate / service_rate) ** servers /
        (factorial(servers) * (1 - rho))
        )

    probability_empty = 1 / probability_empty






    k = np.arange(0, c, 1)

    p_0 = 1 / (
        sum([(c * rho) ** k / factorial(k) for k in k]) +
        (c * rho) ** c / (factorial(c) * (1 - rho))
    )

    l_q = (p_0 * (l / m) ** c * rho) / (factorial(c) * (1 - rho))

    w_q = l_q / l

    return np.nanmax([w_q, 0])


class Station():

    def __init__(self, **kwargs):

        self.cases = kwargs.get('cases', 1) # [-]

        self.rng = kwargs.get('rng', np.random.default_rng())

        self.ports = kwargs.get('ports', 1)

        self.supply_rate = kwargs.get('supply_rate', 400e3)

        self.supply_price = kwargs.get('charge_price', .5 / 3.6e6) # [$/J]

        self.delay = kwargs.get('delay', 0)
        
        self.reliability = kwargs.get('reliability', 1)

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

        self.service = (
            self.clip_norm(self.service_param, self.service_limits) /
            self.expected_charge_rate + self.base_delay
            )

        self.arrival = (
            self.service.mean() / (
                self.chargers * self.clip_norm(self.arrival_param, self.arrival_limits)
                )
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

    def expect(self, vehicle):

        self.delay_time = super_quantile(self.delay_time, vehicle.risk_attitude)

        self.at_least_one_functional_charger = super_quantile(
            self.functional_chargers,
            (1 - vehicle.risk_attitude[1], 1 - vehicle.risk_attitude[0])
            ) > 1

    def update(self, x, vehicle):

        capacity = vehicle.capacity
        target = vehicle.charge_target_soc
        rate = min([self.max_charge_rate, vehicle.charge_rate])

        x = self.time(x, capacity, rate, target)
        x = self.price(x, capacity, rate, target)
        x = self.soc(x, capacity, rate, target)

        return x

    def time(self, x, capacity, rate, target):

        full_charge_time = capacity / rate
        charge_time = full_charge_time * np.clip(target - x['soc'], 0, 1)

        x['time'] += charge_time + self.delay_time

        x['routing_time'] += full_charge_time + self.delay_time

        return x

    def price(self, x, capacity, rate, target):

        price = (
            (capacity * np.clip(target - x['soc'], 0, 1) * self.charge_price)
            )

        x['price'] += price

        return x

    def soc(self, x, capacity, rate, target):

        x['soc'] += np.clip(target - x['soc'], 0, 1) * self.at_least_one_functional_charger

        return x