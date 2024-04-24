import numpy as np
import networkx as nx

from scipy.special import factorial
from scipy.stats import rv_histogram

from .graph import graph_from_nlg
from .utilities import pythagorean, top_n_indices
# from .routing import Charger

def multiply_and_resample(x, y, rng = np.random.default_rng(None)):

	xg, yg = np.atleast_2d(x, y)

	xy = (xg.T @ yg).flatten()

	return rng.choice(xy, size = x.shape, replace = False)

def queuing_time(l, m, c):

	rho = l / (c * m)

	k = np.arange(0, c, 1)

	p_0 = 1 / (
		sum([(c * rho) ** k / factorial(k) for k in k]) +
		(c * rho) ** c / (factorial(c) * (1 - rho))
	)

	l_q = (p_0 * (l / m) ** c * rho) / (factorial(c) * (1 - rho))

	w_q = l_q / l

	return w_q

class Queuing_Time_Distribution():

	def __init__(self, arrival, service, servicers, seed = None, bins = 50, shape = (100, )):

		self.arrival = arrival
		self.service = service
		self.servicers = servicers
		self.seed = seed
		self.bins = bins
		self.shape = shape

		self.Compute()

	def Compute(self):

		if self.servicers == 0:

			self.queuing_time = lambda **kwargs: 0

		else:

			arrival_frequency = 1 / self.arrival.rvs(size = self.shape, random_state = self.seed)
			service_frequency = 1 / self.service.rvs(size = self.shape, random_state = self.seed)

			afg, sfg = np.meshgrid(arrival_frequency, service_frequency, indexing  = 'ij')

			queuing_time_values = queuing_time(afg.flatten(), sfg.flatten(), self.servicers)

			self.queuing_time = rv_histogram(
				np.histogram(queuing_time_values, bins = self.bins)
				).rvs

	def __call__(self, **kwargs):

		return self.queuing_time(**kwargs)

def random_graph_nx(n, p, **kwargs):

	k = kwargs.get('k', None)
	scale = kwargs.get('scale', (1, 1))
	link_speeds = kwargs.get('link_speeds', [1])
	seed = kwargs.get('seed', None)

	rng = np.random.default_rng(seed)

	
	g = nx.fast_gnp_random_graph(n, p, seed = seed)
	d = nx.spring_layout(g, k = k, seed = seed)

	for k, v in g._node.items():

		v['x'] = d[k][0] * scale[0]
		v['y'] = d[k][1] * scale[1]
		v['distance'] = 0
		v['time'] = 0
		v['range'] = 0
		v['price'] = 0

	for s, a in g._adj.items():
		for t, l in a.items():

			link_distance = pythagorean(
				g._node[s]['x'],
				g._node[s]['y'],
				g._node[t]['x'],
				g._node[t]['y'],
			)

			link_time = link_distance / rng.choice(link_speeds)

			l['distance'] = link_distance
			l['time'] = link_time
			l['price'] = 0

	return g

def random_graph(n, **kwargs):

	scale = kwargs.get('scale', (1, 1))
	reference_distance = kwargs.get('reference_distance', 1)
	link_speeds = kwargs.get('link_speeds', [1])
	seed = kwargs.get('seed', None)

	
	rng = np.random.default_rng(seed)

	x = (rng.random(n) - .5) * scale[0]
	y = (rng.random(n) - .5) * scale[1]
	
	nodes = []

	for idx in range(n):

		nodes.append({
			'id': idx,
			'x': x[idx],
			'y': y[idx],
			'distance': 0,
			'time': 0,
			'price': 0,
		})

	links = []

	for idx_s in range(n):
		for idx_t in range(n):

			source = nodes[idx_s]['id']
			target = nodes[idx_t]['id']

			link_distance = pythagorean(
				nodes[idx_s]['x'],
				nodes[idx_s]['y'],
				nodes[idx_t]['x'],
				nodes[idx_t]['y'],
			)

			p = np.exp(-link_distance / reference_distance)
			r = rng.random()

			if r > p:

				continue

			link_time = link_distance / rng.choice(link_speeds)

			links.append({
				'source': source,
				'target': target,
				'distance': link_distance,
				'time': link_time,
				'price': 0,
			})

	return graph_from_nlg({'nodes': nodes, 'links': links})

def random_attribute_assignment(graph, key, values, rng = np.random.default_rng(None)):

	assigned_nodes = list(
		rng.choice(
			list(range(len(graph.nodes))),
			len(values),
			replace = False,
		),
	)

	for idx, node in enumerate(assigned_nodes):

		graph._node[node][key] = values[idx]

	return graph, assigned_nodes

def highly_valent_chargers(graph, n_chargers, charger, seed = None):

	valency = [len(graph._adj[n]) for n in graph.nodes]

	charger_nodes = top_n_indices(valency, n_chargers)

	for node in charger_nodes:

		graph._node[node]['charger'] = charger

	return graph, charger_nodes

def random_origin_destination(graph, n_origins, n_destinations, seed = None):

	rng = np.random.default_rng(seed)

	origin_nodes = rng.choice(graph.nodes, n_origins, replace = False)
	destination_nodes = rng.choice(graph.nodes, n_destinations, replace = False)

	return origin_nodes, destination_nodes