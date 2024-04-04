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

from .progress_bar import ProgressBar
from.rng import Queuing_Time_Distribution

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

def dijkstra(graph, origins, **kwargs):
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
            
            cost = 0

            for key, info in objectives.items():

                # Updating the weighted cost for the path
                cost += info(current_values)

            # print(values['soc'], -super_quantile(-values['soc'], (0, .5)))

            feasible = True

            for key, info in constraints.items():

                # Checking if link traversal is possible
                feasible *= info(current_values)

            # print(feasible)

            if not feasible:

                continue
                
            # Charging if availabe
            if 'functions' in current:

                for key, function in current['functions'].items():

                    function(current_values)

            
            # print(visited)
            # print(cost, visited.get(target, np.array([maxsize])))
            # savings = improvement(cost, visited.get(target, np.array([maxsize])), .00000000005)
            # savings = False
            savings = cost < visited.get(target, np.inf)
            # savings = cost < visited.get(target, np.inf) * 1

            if savings:

                visited[target] = cost

                heappush(heap, (cost, next(c), current_values, target))

                if paths is not None:

                    paths[target] = paths[source] + [target]
        # break

    return path_costs, path_values, paths

def super_quantile(x, risk_attitude, n = 10):
    
    q = np.linspace(risk_attitude[0], risk_attitude[1], n)
    # print(q)
    
    sq = 1/(risk_attitude[1] - risk_attitude[0]) * (np.quantile(x, q) * (q[1] - q[0])).sum()

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

class Charger():

    def __init__(self, vehicle, **kwargs):

        self.vehicle = vehicle
        self.arrival = kwargs.get('arrival', uniform(loc = 600, scale = 3600))
        self.service = kwargs.get(
            'service', norm(loc = 60 * 3.6e6 / 80e3, scale = 15 * 3.6e6 / 80e3)
            )
        self.n = kwargs.get('n', 1)
        self.reliability = kwargs.get('reliability', 1)
        self.energy_price = kwargs.get('energy_price', .5 / 3.6e6)
        self.seed = kwargs.get('seed', None)
        self.rng = np.random.default_rng(self.seed)

        self.populate()

    def populate(self):

        self.qtd = Queuing_Time_Distribution(
            self.arrival, self.service, self.n, seed = self.seed
        )

        self.functions = {
            'time': lambda x: self.time(x),
            'delay': lambda x: self.delay(x),
            'price': lambda x: self.price(x),
            'soc': lambda x: self.soc(x),
        }


    def functionality(self, n):

        return self.rng.random(n) <= 1 - (1 - self.reliability) ** self.n

    def time(self, x):

        # try:

        self.functional = self.functionality(len(x['soc']))
        

        time = (
            (self.vehicle.capacity * (1 - x['soc']) / self.vehicle.rate) *
            self.functional
            )

        x['time'] += time

        # print('b', time)

        # except:

            # print(self.vehicle.__dict__().keys())


        return x

    def delay(self, x):
        # print(self.qtd(size = self.vehicle.n_cases) / 3600 * self.functional)

        time = self.qtd(size = self.vehicle.n_cases) * self.functional

        x['time'] += time

        # print('a', time)

        return x

    def price(self, x):

        price = (
            (self.vehicle.capacity * (1 - x['soc']) * self.energy_price) *
            self.functional
            )

        x['price'] += price

        return x

    def soc(self, x):

        x['soc'] += (1 - x['soc']) * self.functional

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

    def populate_deterministic(self):

        self.objectives = {
            'time': lambda x: x['time'],
        }

        self.constraints = {
            'distance': lambda x: x['distance'] <= self.cutoff
            # 'distance': lambda x: in_range(x['distance'], *self.cutoff)
        }

        self.states = {
            'time': {
                'field': 'time',
                'initial': 0,
                'update': lambda x, v: add_simple(
                    x['time'], v.get('time', 0)),
            },
            'distance': {
                'field': 'distance',
                'initial': 0,
                'update': lambda x, v: add_simple(
                    x['distance'], v.get('distance', 0)),
            },
        }

    def populate_stochastic(self):

        self.objectives = {
            'time': lambda x: super_quantile(x['time'], self.risk_attitude),
        }

        self.constraints = {
            'distance': lambda x: (
                super_quantile(x['distance'], self.risk_attitude) <= self.cutoff
            )
        }

        self.states = {
            'time': {
                'field': 'time',
                'initial': np.array([0.] * self.n_cases),
                'update': lambda x, v: add_simple(
                    x['time'], v.get('time', 0)),
            },
            'distance': {
                'field': 'distance',
                'initial': np.array([0.] * self.n_cases),
                'update': lambda x, v: add_simple(
                    x['distance'], v.get('distance', 0)),
            },
        }

class ConstrainedVehicle(Vehicle):

    def __init__(self, **kwargs):

        self.n_cases = kwargs.get('n_cases', 30) # [-]
        self.capacity = kwargs.get('ess_capacity', 80 * 3.6e6) # [J]
        self.efficiency = kwargs.get('efficiency', 500) # [J/m]
        self.rate = kwargs.get('rate', 80e3) # [W]
        self.initial_soc = kwargs.get('initial_soc', 1) # [-]
        self.max_soc = kwargs.get('max_soc', 1) # [-]
        self.min_soc = kwargs.get('min_soc', .2) # [-]
        self.risk_attitude = kwargs.get('risk_attitude', (0, 1)) # [-]

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

    def populate(self):

        self.objectives = {
            'time': lambda x: super_quantile(x['time'], self.risk_attitude),
        }

        self.constraints = {
            'soc': (
                lambda x: (
                    in_range(
                        super_quantile(x['soc'],
                            (1 - self.risk_attitude[0], 1 - self.risk_attitude[1])),
                        self.min_soc, self.max_soc
                    )
                )
            ),
        }

        self.states = {
            'soc': {
                'field': 'soc',
                'initial': np.array([self.initial_soc] * self.n_cases),
                'update': lambda x, v: add_simple(
                    x['soc'], -v['distance'] * self.efficiency /  self.capacity) ,
            },
            'time': {
                'field': 'time',
                'initial': np.array([0.] * self.n_cases),
                'update': lambda x, v: add_simple(
                    x['time'], v['time']),
            },
            'time_nc': {
                'field': 'time_nc',
                'initial': np.array([0.] * self.n_cases),
                'update': lambda x, v: add_simple(
                    x['time_nc'], v['time']),
            },
            'distance': {
                'field': 'distance',
                'initial': np.array([0.] * self.n_cases),
                'update': lambda x, v: add_simple(
                    x['distance'], v['distance']),
            },
            'price': {
                'field': 'price',
                'initial': np.array([0.] * self.n_cases),
                'update': lambda x, v: add_simple(x['price'], v['price']),
            },
        }