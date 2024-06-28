import numpy as np

from .graph import graph_from_nlg
from .utilities import pythagorean

def random_completely_connected_graph(n, m, **kwargs):

	n_nodes = n + m

	scale = kwargs.get('scale', (1, 1))
	speeds = kwargs.get('speeds', [1])
	chargers = kwargs.get('chargers', [1])
	seed = kwargs.get('seed', None)
	labels = kwargs.get('labels', ('place', 'station'))

	rng = np.random.default_rng(seed)

	x = (rng.random(n_nodes) - .5) * scale[0]
	y = (rng.random(n_nodes) - .5) * scale[1]
	
	nodes = []

	for idx in range(n):

		nodes.append({
			'id': f'{labels[0]}_{idx}',
			'x': x[idx],
			'y': y[idx],
			'distance': 0,
			'time': 0,
			'price': 0,
			'type': 'place',
		})

	for idx in range(n, n_nodes):

		nodes.append({
			'id': f'{labels[1]}_{idx - n}',
			'x': x[idx],
			'y': y[idx],
			'distance': 0,
			'time': 0,
			'price': 0,
			'n_dcfc': rng.choice(chargers),
			'type': 'station',
		})

	links = []

	for idx_s in range(n_nodes):
		for idx_t in range(n_nodes):

			source = nodes[idx_s]['id']
			target = nodes[idx_t]['id']

			distance = pythagorean(
				nodes[idx_s]['x'],
				nodes[idx_s]['y'],
				nodes[idx_t]['x'],
				nodes[idx_t]['y'],
			)

			time = distance / rng.choice(speeds)

			links.append({
				'source': source,
				'target': target,
				'distance': distance,
				'time': time,
				'price': 0,
			})

	return graph_from_nlg({'nodes': nodes, 'links': links}, **kwargs.get('graph', {}))