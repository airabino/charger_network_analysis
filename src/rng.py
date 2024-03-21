import numpy as np

from .graph import graph_from_nlg
from .utilities import pythagorean, top_n_indices
from .routing import Charger

def random_graph(n, **kwargs):

	scale = kwargs.get('scale', (1, 1))
	reference_distance = kwargs.get('reference_distance', 1)
	link_bounds = kwargs.get('link_bounds', (0, np.inf))
	link_speeds = kwargs.get('link_speeds', [1])
	seed = kwargs.get('seed', None)
	range_multiplier = kwargs.get('range_multiplier', -1)

	
	rng = np.random.default_rng(seed)

	x = rng.random(n) * scale[0]
	y = rng.random(n) * scale[1]
	
	nodes = []

	for idx in range(n):

		nodes.append({
			'id': idx,
			'x': x[idx],
			'y': y[idx],
			'distance': 0,
			'time': 0,
			'range': 0,
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
			# if r <= p:
				# print(idx_s, idx_t, p, r, link_distance)

			dont_add_link = np.any((
				(link_distance < link_bounds[0]),
				(link_distance > link_bounds[1]),
				r > p,
				))

			if dont_add_link:

				continue

			# print(idx_s, idx_t, p, r, link_distance)

			link_time = link_distance / rng.choice(link_speeds)

			links.append({
				'source': source,
				'target': target,
				'distance': link_distance,
				'time': link_time,
				'range': link_distance * range_multiplier,
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