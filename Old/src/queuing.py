import time
import numpy as np

from heapq import heappop, heappush
from itertools import count

class Server():

	def __init__(self, **kwargs):

		self.service_rate = kwargs.get('service_rate', lambda rng: 1) # [u/s]

		self.customer = None

		self.status = 'vacant'

	def start(self, customer):

		self.customer = customer

		self.status = 'occupied'

	def step(self):

		status = self.customer.step(self.service_rate)

		if status == 'served':

			self.status = 'complete'

	def finish(self):

		self.status = 'vacant'

		return self.customer

class Demand():

	def __init__(self, **kwargs):

		self.rng = kwargs.get('rng', np.random.default_rng())

		self.spawn_rate = kwargs.get('spawn_rate', 1 / 1000) # [v/s]

		self.capacity = kwargs.get('capacity', lambda rng: 1) # [u]
		
	def spawn(self):

		if rng.random() <= self.spawn_rate:

			return Customer(self.capacity())

class Customer():

	def __init__(self, **kwargs):

		self.capacity = kwargs.get('capacity', 1) # [u]

		self.level = 0

		self.status = 'queuing'
	
	def step(self, service_rate):

		self.status = 'receiving'

		self.level += service_rate

		if self.level >= self.capacity:

			self.status = 'complete'

class System():

	def __init__(self, servers, demand):

		self.servers = servers
		self.demand = demand

	def simulate(self, steps = 1000):

		counter = count

		queue = []
		served = []

		status = {
			'in_queue': np.zeros(steps),
			'in_service': np.zeros(steps),
			'sered': np.zeros(steps),
		}

		for idx in range(steps):

			# Creating new customers
			customer = self.demand.spawn()

			if customer is not None:

				heappush(queue, (next(counter), customer))

			# Advancing servers
			for server in servers:

				# Simulating one step
				server.step()

				# Moving served customers
				if server.status == 'complete':

					customer = server.finish()

					served.append(customer)

				# Moving customers to vacant servers
				if (server.status) == 'vacant' and queue:

					_, customer = heappop(queue)

					server.serve(customer)

		return queue, served, status



