import os
import sys
import time
import matplotlib
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

from cycler import cycler

from .graph import subgraph

from scipy.stats._continuous_distns import _distn_names

default_prop_cycle = matplotlib.rcParamsDefault['axes.prop_cycle'].by_key()['color'].copy()

colormaps={
	'day_night': ["#e6df44", "#f0810f", "#063852", "#011a27"],
	'beach_house': ["#d5c9b1", "#e05858", "#bfdccf", "#5f968e"],
	'autumn': ["#db9501", "#c05805", "#6e6702", "#2e2300"],
	'ocean': ["#003b46", "#07575b", "#66a5ad", "#c4dfe6"],
	'forest': ["#7d4427", "#a2c523", "#486b00", "#2e4600"],
	'aqua': ["#004d47", "#128277", "#52958b", "#b9c4c9"],
	'field': ["#5a5f37", "#fffae1", "#524a3a", "#919636"],
	'misty': ["#04202c", "#304040", "#5b7065", "#c9d1c8"],
	'greens': ["#265c00", "#68a225", "#b3de81", "#fdffff"],
	'citroen': ["#b38540", "#563e20", "#7e7b15", "#ebdf00"],
	'blues': ["#1e1f26", "#283655",  "#4d648d", "#d0e1f9"],
	'dusk': ["#363237", "#2d4262", "#73605b", "#d09683"],
	'ice': ["#1995ad", "#a1d6e2", "#bcbabe", "#f1f1f2"],
	'csu': ["#1e4d2b", "#c8c372"],
	'ucd': ['#022851', '#ffbf00'],
	'incose': ["#f2606b", "#ffdf79", "#c6e2b1", "#509bcf"],
	'sae': ["#01a0e9", "#005195", "#cacac8", "#9a9b9d", "#616265"],
	'trb': ["#82212a", "#999999", "#181818"],
	'default_prop_cycle': default_prop_cycle,
}

default_route_tree_kwargs = {
	'destinations_kw': {
		'node_field': 'value',
		'scatter': {
			's': 300,
			'ec': 'k',
			'lw': 2,
			# 'marker': (8, 1, 0),
			'zorder': 3,
			'label': 'Destinations',
		},
		'colorbar': {
			'label': 'Time to Destination [h]'
		}
	},
	'stations_kw': {
		# 'node_field': 'value',
		'scatter': {
			's': 200,
			'fc': 'none',
			'ec': 'k',
			'zorder': 1,
			'marker': (4, 1, 0),
			'label': 'Unused Stations',
		},
	},
	'stations_used_kw': {
		# 'node_field': 'value',
		'scatter': {
			's': 300,
			'fc': 'magenta',
			'ec': 'k',
			'lw': 2,
			'zorder': 2,
			'marker': (4, 1, 0),
			'label': 'Used Stations',
		},
	},
	'edge_kw': {
		'color': 'k',
		'lw': 1.5,
		'zorder': 0,
	}
}

def plot_route_tree(ax, graph, values = {}, paths = {}, destinations = [], **kwargs):
	'''
	plots a route tree from an origin to destinations with refueling stations
	'''

	stations_used = []

	# Plotting path edges
	for path in paths.values():

		stations_used.extend(path[1:-1])

		x = [graph._node[n]['x'] for n in path]
		y = [graph._node[n]['y'] for n in path]

		ax.plot(x, y, **kwargs.get('edge_kw', {}))

	# Adding value field to graph
	for source, node in graph._node.items():

		node['value'] = values.get(source, np.nan)

	# Making subgraphs
	nodes = list(graph.nodes)
	stations = [n for n in nodes if n not in destinations]
	stations = [n for n in stations if n not in stations_used]

	# Plotting destinations
	if kwargs.get('show_destinations', True):

		destinations = subgraph(graph, destinations)
	
		plot_graph(
			destinations, ax = ax, show_links = False, **kwargs.get('destinations_kw', {})
			)

	# Plotting stations
	if kwargs.get('show_unused_stations', True):

		stations = subgraph(graph, stations)
	
		plot_graph(
			stations, ax = ax, show_links = False, **kwargs.get('stations_kw', {})
			)

	# Plotting stations used
	if kwargs.get('show_used_stations', True):

		stations_used = subgraph(graph, stations_used)
	
		plot_graph(
			stations_used, ax = ax, show_links = False, **kwargs.get('stations_used_kw', {})
			)

def colormap(colors, reverse = False):

	if type(colors) == str:

		if colors in colormaps.keys():

			colors_list = colormaps[colors]

			if reverse:

				colors_list = np.flip(colors_list)

			colormap_out = LinearSegmentedColormap.from_list(
				'custom', colors_list, N = 256)

		else:

			colormap_out = matplotlib.cm.get_cmap(colors)

	else:

		colormap_out = LinearSegmentedColormap.from_list(
			'custom', colors, N = 256)

	return colormap_out

def add_node_field(graph, field, values):

	for idx, key in enumerate(graph._node.keys()):

		graph._node[key][field] = values[idx]

	return graph

def plot_graph(graph, ax = None, **kwargs):

	cmap = kwargs.get('cmap', colormap('viridis'))
	node_field = kwargs.get('node_field', None)
	link_field = kwargs.get('link_field', None)
	show_links = kwargs.get('show_links', True)
	show_colorbar = kwargs.get('show_colorbar', False)
	colorbar_kw = kwargs.get('colorbar', {})
	
	return_fig = False

	if ax is None:

		fig, ax = plt.subplots(**kwargs.get('figure', {}))
		return_fig = True

	coords = np.array([[node['x'], node['y']] for node in graph._node.values()])

	scatter_kw = kwargs.get('scatter', {})

	if node_field is not None:

		values = np.array([v.get(node_field, np.nan) for v in graph._node.values()])

		indices = np.argsort(values)

		values = values[indices]
		coords = coords[indices]

		vmin = scatter_kw.get('vmin', np.nanmin(values))
		vmax = scatter_kw.get('vmax', np.nanmax(values))

		if vmin == vmax:

			values_norm = values

		else:

			values_norm = (
					(values - vmin) / (vmax - vmin)
					)

		scatter_kw['color'] = cmap(values_norm)

	sc = ax.scatter(
		coords[:, 0], coords[:, 1], **scatter_kw
		)

	if show_links:

		dx = []
		dy = []

		for source in graph._adj.keys():

			for target in graph._adj[source].keys():

				dx.append([graph._node[source]['x'], graph._node[target]['x']])
				dy.append([graph._node[source]['y'], graph._node[target]['y']])

		ax.plot(np.array(dx).T, np.array(dy).T, **kwargs.get('plot', {}))

	ax.set(**kwargs.get('axes', {}))

	if show_colorbar or colorbar_kw:

		norm = matplotlib.colors.Normalize(
			vmin = scatter_kw.get('vmin', np.nanmin(values)),
			vmax = scatter_kw.get('vmax', np.nanmax(values))
			) 

		sm = matplotlib.cm.ScalarMappable(cmap = cmap, norm = norm)    
		sm.set_array([])

		plt.colorbar(sm, ax = ax, **kwargs.get('colorbar', {})) 

	if return_fig:

		return fig