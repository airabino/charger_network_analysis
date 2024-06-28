import time
import numpy as np

from heapq import heappop, heappush
from itertools import count
from scipy.stats import rv_histogram, norm
from scipy.special import factorial

class Server():

    def __init__(self, **kwargs):

        self.rng = kwargs.get('rng', np.random.default_rng())

        self.service_rate = kwargs.get('service_rate', lambda rng: 1)(self.rng) # [u/s]

        self.customer = None

        self.status = 'vacant'

    def start(self, customer):

        self.customer = customer

        self.status = 'occupied'

    def step(self, step):

        self.customer.step(step)

        status = self.customer.receive(self.service_rate, step)

        if status == 'complete':

            self.status = 'complete'

    def finish(self):

        self.status = 'vacant'

        return self.customer

class Demand():

    def __init__(self, **kwargs):

        self.rng = kwargs.get('rng', np.random.default_rng())

        self.max_length = kwargs.get('max_length', np.inf)

        self.inter_arrival = kwargs.get('inter_arrival', lambda rng: rng.exponential(1))

        self.interval = self.inter_arrival(self.rng)

        self.capacity = kwargs.get('capacity', lambda rng: 1) # [u]

        self.initial = kwargs.get('initial', 0)

        self.initial_customers = (
            [Customer(capacity = self.capacity(self.rng)) for idx in range(self.initial)]
            )
        
    def spawn(self, step):

        self.interval -= step

        if self.interval <= 0:

            self.interval = self.inter_arrival(self.rng)

            return Customer(capacity = self.capacity(self.rng))

class Customer():

    def __init__(self, **kwargs):

        self.capacity = kwargs.get('capacity', 1) # [u]

        self.level = 0

        self.steps = 0
        self.steps_service = 0

        self.status = 'queuing'

    def step(self, step):

        self.steps += step
    
    def receive(self, service_rate, step):

        self.status = 'receiving'

        self.level += service_rate * step

        self.steps_service += step

        if self.level >= self.capacity:

            self.status = 'complete'

            # print( (self.level - self.capacity) * step)

            self.steps_service += (self.level - self.capacity) / self.capacity  * step

        return self.status

class System():

    def __init__(self, servers, demand):

        self.servers = servers
        self.demand = demand

    def simulate(self, steps = 1000, step = 1):

        counter = count()

        queue = []
        served = []

        for customer in self.demand.initial_customers:

            heappush(queue, (next(counter), customer))

        in_queue = 0
        in_service = 0
        in_served = 0

        status = {
            'in_queue': [],
            'in_service': [],
            'in_served': [],
        }

        for idx in range(0, steps, step):

            # Advancing customers
            for customer in queue:

                customer[1].step(step)

            # Creating new customers
            customer = self.demand.spawn(step)

            if (customer is not None) and (len(queue) <= self.demand.max_length - 1):

                heappush(queue, (next(counter), customer))

                in_queue += 1

            # Advancing servers
            for server in self.servers:

                # Simulating one step
                if server.status == 'occupied':

                    server.step(step)

                # Moving served customers
                if server.status == 'complete':

                    customer = server.finish()

                    served.append(customer)

                    in_service -= 1
                    in_served += 1

                # Moving customers to vacant servers
                if (server.status) == 'vacant' and queue:

                    _, customer = heappop(queue)

                    server.start(customer)

                    in_queue -= 1
                    in_service += 1

            status['in_queue'].append(in_queue)
            status['in_service'].append(in_service)
            status['in_served'].append(in_served)

        return queue, served, status

def simulated_queuing_time_distribution(**kwargs):

    servers = [Server(**kwargs.get('server', {})) for idx in range(kwargs.get('n', 1))]

    demand = Demand(**kwargs.get('demand', {}))

    system = System(servers, demand)

    _, served, _ = system.simulate(**kwargs.get('simulation', {}))

    queue_steps = np.array([customer.steps - customer.steps_service for customer in served])

    return rv_histogram(np.histogram(queue_steps, **kwargs.get('histogram', {})))

def mmc_queue(arrival_rate, service_rate, servicers, max_time = np.inf):

    rho = arrival_rate / (service_rate * servicers)

    probability_empty_denominator = 0

    for k in range(servicers):

        probability_empty_denominator += (servicers * rho) ** k / factorial(k)

    probability_empty_denominator += (
        (servicers * rho) ** servicers / factorial(servicers) / (1 - rho)
        )

    probability_empty = 1 / probability_empty_denominator

    waiting_time = (
        probability_empty * rho * (servicers * rho) ** servicers /
        (arrival_rate * (1 - rho) ** 2 * factorial(servicers))
        )

    return min([np.nanmax([waiting_time, 0]), max_time])

def queuing_time_distribution(n, rho, power, **kwargs):

    rho = rho[rho <= .99]

    service_mu = kwargs.get('service_mu', 30 * 60)
    service_sigma = kwargs.get('service_sigma', 10 * 60)

    service_rate_distribution = (
        lambda rho: 1 / np.clip(norm(service_mu, service_sigma).ppf(rho), 1, np.inf)
        )

    service_rate = service_rate_distribution(rho)

    waiting_time = np.zeros(len(rho))
    
    for idx in range(len(rho)):

        arrival_rate = rho[idx] * service_rate[idx] * n
        # print(n)

        waiting_time[idx] = mmc_queue(
            arrival_rate, service_rate[idx], n,
            max_time = kwargs.get('max_time', np.inf),
        )

    # print(n, max(waiting_time))

    dist = rv_histogram(
        np.histogram(
            waiting_time, **kwargs.get(
                'histogram', {'bins': np.arange(0, max([max(waiting_time), 60]) + 60, 60)}
                )
            )
        )

    return dist