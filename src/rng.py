import numpy as np

from .graph import graph_from_nlg
from .utilities import pythagorean

def random_graph(n, scale, s = 1, link_bounds = (0, np.inf), link_speeds = [1], seed = None):
	
	rng = np.random.default_rng(seed)

	x = rng.random(n) * scale[0]
	y = rng.random(n) * scale[1]
	
	nodes = []

	for idx in range(n):

		nodes.append({
			'id': idx,
			'x': x[idx],
			'y': y[idx],
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

			p = np.exp(-link_distance / s)
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
				'range': -link_distance,
			})

	return graph_from_nlg({'nodes': nodes, 'links': links})

def random_chargers(graph, n_chargers, charger, seed = None):

	rng = np.random.default_rng(seed)

	# Adding cost functions to nodes
	charger_nodes = list(
		rng.choice(
			list(range(len(graph.nodes))),
			n_chargers,
			replace = False,
		),
	)

	for node in charger_nodes:

		graph._node[node]['charger'] = charger

	return graph, charger_nodes

def random_origin_destination(graph, n_origins, n_destinations, seed = None):

	rng = np.random.default_rng(seed)

	origin_nodes = rng.choice(graph.nodes, n_origins, replace = False)
	destination_nodes = rng.choice(graph.nodes, n_destinations, replace = False)

	return origin_nodes, destination_nodes