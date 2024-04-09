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

def multiply_and_resample_factorial(x, y, rng = np.random.default_rng(None)):

    xg, yg = np.atleast_2d(x, y)

    xy = (xg.T @ yg).flatten()

    return rng.choice(xy, size = x.shape, replace = False)

def add_and_resample_factorial(x, y, rng = np.random.default_rng(None)):

    xg, yg = np.atleast_2d(x, y)

    xy = (xg.T + yg).flatten()

    return rng.choice(xy, size = x.shape, replace = False)

def multiply_and_resample(x, y, rng = np.random.default_rng(None)):

    xy = x * rng.permutation(np.atleast_1d(y))

    return xy

def add_and_resample(x, y, rng = np.random.default_rng(None)):

    xy = x + rng.permutation(np.atleast_1d(y))

    return xy

def add_simple(x, y, **kwargs):

    xy = x + y

    return xy

def dijkstra_dict(graph, origins, **kwargs):
    """
    Uses Dijkstra's algorithm to find weighted shortest paths

    Code is based on NetworkX shortest_paths.weighted._dijkstra_multisource

    This implementation of Dijkstra's method is designed for high flexibility
    at some cost to efficiency. Specifically, this function implements Stochastic
    Cost with Risk Allowance Minimization Dijkstra (SCRAM-D) routing. As such
    this function allows for an optimal route to be computed for probabilistic
    node/link costs by tracking N scenarios in parallel with randomly sampled
    costs and minimizing the expectation of cost subject to constraints which
    may also be based on cost expectation. Additionally, nodes amy contain Charger
    objects which serve to reset a given state to a pre-determined value and
    may effect other states.

    Example - Battery Electric Vehicle (BEV) routing:

    graph - Graph or DiGraph containing a home location, several destinations,
    and M chargers of varying reliability and links for all objects less than
    300 km apart.

    origins - [home]

    destinations - [Yellowstone Park, Yosemite Park, Grand Canyon]

    states - {
        'distance': {
            'field': 'distance',
            'initial': [0] * n_cases,
            'update': lambda x, v: [xi + v for xi in x],
            'cost': lambda x: 0,
        },
        'price': {
            'field': 'price',
            'initial': [0] * n_cases,
            'update': lambda x, v: [xi + v for xi in x],
            'cost': lambda x: 0,
        },
        'time': {
            'field': 'time',
            'initial': [0] * n_cases,
            'update': lambda x, v: [xi + v for xi in x],
            'cost': lambda x: src.utilities.super_quantile(x, risk_tolerance),
        },
    }

    constraints - {
        'range': {
            'field': 'range',
            'initial': [vehicle_range] * n_cases,
            'update': lambda x, v: [xi + v for xi in x],
            'feasible': lambda x: src.utilities.super_quantile(x, risk_tolerance) > min_range,
        },
    }

    Parameters
    ----------
    graph: a NetworkX Graph or DiGraph

    origins: non-empty iterable of nodes
        Starting nodes for paths. If origins is an iterable containing
        a single node, then all paths computed by this function will
        start from that node. If there are two or more nodes in origins
        the shortest path to a given destination may begin from any one
        of the start nodes.

    destinations: iterable of nodes - optionally empty
        Ending nodes for path. If destinations is not empty then search
        continues until all reachable destinations are visited. If destinations
        is empty then the search continues until all reachable nodes are visited.

    states: dictionary with the below fields:
        'field': The relevant node/link property for integration
        'initial': Initial values for integration
        'update': Function which updates path values for nodes/links
        'cost': Function for computing a cost expectation from values

        states are used to compute the cost expectation

    constraints: dictionary with the below fields:
        'field': The relevant node/link property for integration
        'initinal': Iitial values for integration
        'update': Function which updates path values for nodes/links
        'feasible': Function which returns a Boolean for feasibility from values

    parameters: dictionary with the below fields:
        'field': The relevant node/link property for integration
        'initinal': Iitial values for integration
        'update': Function which updates path values for nodes/links

        parameters are NOT used to compute the cost expectation

    return_paths: Boolean
        Boolean whether or not to compute paths dictionary. If False None
        is returned for the paths output. COMPUTING PATHS WILL INCREASE RUN-TIME.

    Returns
    -------

    path_costs : dictionary
        Path cost expectations.

    path_values : dictionary
        Path objective values.

    paths : dictionary
        Dictionary containing ordered lists of nodes passed on shortest
        path between the origin node and other nodes. If return_paths == False
        then None will be returned.
    """

    destinations = kwargs.get('destinations', [])
    states = kwargs.get('states', {'network_distance': {'update': lambda x: x+1}})
    constraints = kwargs.get('constraints', [])
    objectives = kwargs.get('objectives', [])
    return_paths = kwargs.get('return_paths', False)

    if return_paths:

        paths = {origin: [origin] for origin in origins}

    else:

        paths = None

    adjacency = graph._adj

    path_values = {}  # dictionary of costs for paths

    path_costs = {} # dictionary of objective values for paths

    visited = {} # dictionary of costs-to-reach for nodes

    destinations_visited = 0

    if len(destinations) == 0:

        # If no destinations are provided then search all nodes
        destinations_to_visit = maxsize

    else:

        #If destinations are provided then search until all are seen
        destinations_to_visit = len(destinations)

    c = count() # use the count c to avoid comparing nodes (may not be able to)
    heap = [] # heap is heapq with 3-tuples (cost, c, node)

    for origin in origins:

        visited[origin] = np.array([0]) # Source is seen at the start of iteration and at 0 cost

        # Adding the source tuple to the heap (initial cost, count, id)
        values = {}

        for key, info in states.items():

            values[key] = info['initial']

        heappush(heap, (0, next(c), values, origin))

    while heap: # Iterating while there are accessible unseen nodes

        # Popping the lowest cost unseen node from the heap
        (cost, _, values, source) = heappop(heap)
        print(source, end = '\r')

        if source in path_values:

            continue  # already searched this node.

        path_values[source] = values
        path_costs[source] = cost

        # Checking if the current source is a search target
        # If all targets are reached then the search is terminated
        if source in destinations:

            destinations_visited += 1

        if destinations_visited >= destinations_to_visit:

            break

        # print('b')

        # Iterating through the current source node's adjacency
        for target, link in adjacency[source].items():

            current = graph._node[target]

            current_values = deepcopy(values)            

            for key, info in states.items():

                # Updating objective for link
                current_values[key] = info['update'](
                    current_values, link
                    )

                # Adding the target node cost
                current_values[key] = info['update'](
                    current_values, current
                    )

            # cost = objectives(current_values)
            
            cost = 0

            for key, info in objectives.items():

                # Updating the weighted cost for the path
                cost += info(current_values)

            feasible = True

            for key, info in constraints.items():

                # Checking if link traversal is possible
                feasible *= info(current_values, current)

            if not feasible:

                continue
                
            # Charging if availabe
            if 'functions' in current:

                for key, function in current['functions'].items():

                    function(current_values)

            savings = cost < visited.get(target, np.inf)

            if savings:

                visited[target] = cost

                heappush(heap, (cost, next(c), current_values, target))

                if paths is not None:

                    paths[target] = paths[source] + [target]
        # break

    return path_costs, path_values, paths

def dijkstra(graph, origins, **kwargs):

    destinations = kwargs.get('destinations', [])
    states = kwargs.get('states', {'network_distance': {'update': lambda x: x+1}})
    constraints = kwargs.get('constraints', [])
    objectives = kwargs.get('objectives', [])
    return_paths = kwargs.get('return_paths', False)

    if return_paths:

        paths = {origin: [origin] for origin in origins}

    else:

        paths = None

    adjacency = graph._adj

    path_values = {}  # dictionary of costs for paths

    path_costs = {} # dictionary of objective values for paths

    visited = {} # dictionary of costs-to-reach for nodes

    destinations_visited = 0

    if len(destinations) == 0:

        # If no destinations are provided then search all nodes
        destinations_to_visit = maxsize

    else:

        #If destinations are provided then search until all are seen
        destinations_to_visit = len(destinations)

    c = count() # use the count c to avoid comparing nodes (may not be able to)
    heap = [] # heap is heapq with 3-tuples (cost, c, node)

    for origin in origins:

        # Source is seen at the start of iteration and at 0 cost
        visited[origin] = np.array([0])

        # Adding the source tuple to the heap (initial cost, count, id)
        values = {}

        values = states['initial']

        heappush(heap, (0, next(c), values, origin))

    while heap: # Iterating while there are accessible unseen nodes

        # Popping the lowest cost unseen node from the heap
        (cost, _, values, source) = heappop(heap)
        print(source, end = '\r')

        if source in path_values:

            continue  # already searched this node.

        path_values[source] = values
        path_costs[source] = cost

        # Checking if the current source is a search target
        # If all targets are reached then the search is terminated
        if source in destinations:

            destinations_visited += 1

        if destinations_visited >= destinations_to_visit:

            break

        # Charging if availabe
        values = graph._node[source].get('update', lambda x: x)(values)

        # Iterating through the current source node's adjacency
        for target, link in adjacency[source].items():

            node = graph._node[target]

            current_values = deepcopy(values)            

            # Updating objective for link
            current_values = states['update'](
                current_values, link
                )

            # Adding the target node cost
            current_values = states['update'](
                current_values, node
                )

            # Updating the weighted cost for the path
            cost = objectives(current_values)

            savings = cost < visited.get(target, np.inf)

            if savings:

                # Checking if link traversal is possible
                feasible = constraints(current_values, node)

                if feasible:

                    visited[target] = cost

                    heappush(heap, (cost, next(c), current_values, target))

                    if paths is not None:

                        paths[target] = paths[source] + [target]

    return path_costs, path_values, paths

def super_quantile(x, risk_attitude, n = 10):
    
    q = np.linspace(risk_attitude[0], risk_attitude[1], n)
    # print(q)
    
    sq = 1/(risk_attitude[1] - risk_attitude[0]) * (np.quantile(x, q) * (q[1] - q[0])).sum()

    # sq = x.mean() + x.std()

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

def in_range(x, lower, upper):

    return (x >= lower) & (x <= upper)

class Station():

    def __init__(self, vehicle, **kwargs):

        self.vehicle = vehicle
        self.arrival = kwargs.get('arrival', uniform(loc = 600, scale = 3600))
        self.rate = kwargs.get('rate', 80e3)
        self.service = kwargs.get(
            'service', norm(loc = 60 * 3.6e6 / self.rate, scale = 15 * 3.6e6 / self.rate)
            )
        self.chargers = kwargs.get('servicers', 1)
        self.reliability = kwargs.get('reliability', 1)
        self.energy_price = kwargs.get('energy_price', .5 / 3.6e6)
        self.seed = kwargs.get('seed', None)
        self.rng = np.random.default_rng(self.seed)

        self.populate()

    def populate(self):

        self.functional_chargers = self.functionality()
        self.at_least_one_functional_charger = self.functional_chargers > 0

        self.delay_time = np.zeros(self.vehicle.n_cases)

        for idx in range(self.vehicle.n_cases):

            self.delay_time[idx] = self.queuing_time(
                1 / self.arrival.rvs(), 1 / self.service.rvs(), self.functional_chargers[idx]
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
            (self.vehicle.capacity * (1 - x['soc']) / self.vehicle.rate + self.delay_time) *
            self.at_least_one_functional_charger
            )

        x['time'] += time

        return x

    def price(self, x):

        price = (
            (self.vehicle.capacity * (1 - x['soc']) * self.energy_price) *
            self.at_least_one_functional_charger
            )

        x['price'] += price

        return x

    def soc(self, x):

        x['soc'] += (1 - x['soc']) * self.at_least_one_functional_charger

        return x

class Vehicle():

    def __init__(self, **kwargs):

        self.n_cases = kwargs.get('n_cases', 1) # [-]
        self.risk_attitude = kwargs.get('risk_attitude', (0, 1)) # [-]
        self.cutoff = kwargs.get('cutoff', np.inf) # [m]

        if self.n_cases == 1:

            self.populate_deterministic()

        else:

            self.populate_stochastic()

    def all_pairs(self, graph, nodes = [], **kwargs):

        expectations_all_pairs = {}

        if not nodes:

            nodes = list(graph.nodes)

        for node in ProgressBar(nodes, **kwargs.get('progress_bar', {})):

            expectations, _, _ = self.routes(graph, [node], destinations = nodes)

            expectations_all_pairs[node] = (
                {key: val for key, val in expectations.items() if key in nodes}
                )

        return expectations_all_pairs

    def routes(self, graph, origins, destinations = [], return_paths = False):

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

        x['time'] += v['time']
        x['distance'] += v['distance']

        return x

    def populate_deterministic(self):

        self.objectives = lambda x: x['time'],

        self.constraints = lambda x: x['distance'] <= self.cutoff

        self.states = {
            'initial': {
                'time': 0,
                'distance': 0,
            },
            'update': lambda x, v: self.state_update(x, v),
        }

    def populate_stochastic(self):

        self.objectives = lambda x: super_quantile(x['time'], self.risk_attitude),

        self.constraints = lambda x: (
                super_quantile(x['distance'], self.risk_attitude) <= self.cutoff
            )

        self.states = {
            'initial': {
                'time': np.array([0.] * self.n_cases),
                'distance': np.array([0.] * self.n_cases),
            },
            'update': lambda x, v: self.state_update(x, v),
        }

class ConstrainedVehicle(Vehicle):

    def __init__(self, **kwargs):

        self.n_cases = kwargs.get('n_cases', 30) # [-]
        self.capacity = kwargs.get('ess_capacity', 80 * 3.6e6) # [J]
        self.efficiency = kwargs.get('efficiency', 500) # [J/m]
        self.rate = kwargs.get('rate', 80e3) # [W]
        self.initial_soc = kwargs.get('initial_soc', 1.) # [-]
        self.max_soc = kwargs.get('max_soc', 1.) # [-]
        self.min_soc = kwargs.get('min_soc', .2) # [-]
        self.risk_attitude = kwargs.get('risk_attitude', (0, 1)) # [-]

        if self.n_cases > 1:

            self.expectation_function = super_quantile

        else:

            self.expectation_function = lambda x, a: x[0]

        self.populate()

    def routes(self, graph, origins, destinations = [], return_paths = False):

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

        return feasible

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