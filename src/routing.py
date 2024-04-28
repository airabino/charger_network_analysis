import time
import numpy as np

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
        self.charge_price = kwargs.get('charge_price', .5 / 3.6e6) # [$/J]
        self.soc_bounds = kwargs.get('soc_bounds', (0, 1)) # ([-], [-])

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
        # print(link)

        if not in_range(link['distance'], 0, self.range):

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

            delta_soc = self.soc_bounds[1] - updated_values['soc']

            updated_values['soc'] += delta_soc
            updated_values['time'] += (
                delta_soc * self.capacity / self.charge_rate + self.charge_time_penalty
                )
            updated_values['price'] += delta_soc * self.capacity * self.charge_price

        return updated_values, feasible

    def compare(self, values, approximation):

        return values['time'], values['time'] < approximation['time']

class StochasticVehicle():

    def __init__(self, **kwargs):

        self.cases = kwargs.get('cases', 30) # [-]
        self.capacity = kwargs.get('capacity', 80 * 3.6e6) # [J]
        self.efficiency = kwargs.get('efficiency', 550) # [J/m]
        self.charge_rate = kwargs.get('charge_rate', 80e3) # [W]
        self.charge_price = kwargs.get('charge_price', .5 / 3.6e6) # [$/J]
        self.soc_bounds = kwargs.get('soc_bounds', (0, 1)) # ([-], [-])
        self.max_charge_start_soc = kwargs.get('max_charge_start_soc', 1) # [-]
        self.risk_attitude = kwargs.get('risk_attitude', (0, 1)) # ([-], [-])

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

        feasible = in_range(self.expectation(updated_values['soc']), *self.soc_bounds)

        if node['type'] == 'station':

            updated_values = node['station'].update(updated_values)

        return updated_values, feasible

    def compare(self, values, comparison):

        return (
            self.expectation(values['time']), 
            self.expectation(values['time']) < self.expectation(comparison['time'])
            )

class StochasticStation():

    def __init__(self, vehicle, **kwargs):

        self.vehicle = vehicle

        self.seed = kwargs.get('seed', None)
        self.rng = kwargs.get('rng', np.random.default_rng())
        self.chargers = kwargs.get('chargers', 1)
        self.charge_rate = kwargs.get('charge_rate', 80e3)

        self.charge_rate = min([self.charge_rate, self.vehicle.charge_rate])

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

        return np.clip(norm(*param).rvs(self.vehicle.cases), *limits)

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

        self.delay_time = np.zeros(self.vehicle.cases)

        for idx in range(self.vehicle.cases):

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

        rn = self.rng.random(size = (self.chargers, self.vehicle.cases))

        return (rn <= self.reliability).sum(axis = 0)

    def time(self, x):
        
        time = (
            (self.vehicle.capacity * (1 - x['soc']) /
                self.charge_rate +
                self.delay_time)
            )

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