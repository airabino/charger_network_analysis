import json
import momepy
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx

from .graph import graph_from_shapefile, dataframe_from_xlsx

def entities_graph():

	place_shapefile = 'Data/Places/tl_2023_06_place.shp'

	node_attributes = {
		'name': 'lambda n: n["NAME"]',
		'class': 'lambda n: n["CLASSFP"]',
		'geoid': 'lambda n: n["GEOID"]',
		'x': 'lambda n: n["geometry"].centroid.x',
		'y': 'lambda n: n["geometry"].centroid.y',
	}

	graph_place = graph_from_shapefile(
		place_shapefile,
		node_attributes,
		contains_links = False
		)

	place_excel = 'Data/Places/SUB-IP-EST2022-POP-06.xlsx'

	kw = {
		'skiprows': 3,
		'skipfooter': 5,
		'usecols': ['Unnamed: 0', 'Unnamed: 1'],
	}

	df_place = dataframe_from_xlsx(place_excel, **kw)

	df_place = df_place.rename(
		columns = {
			'Unnamed: 0': 'place',
			'Unnamed: 1': 'population'
		}
	)

	df_place['place'] = df_place['place'].apply(lambda s: s.replace(' city, California', ''))
	df_place['place'] = df_place['place'].apply(lambda s: s.replace(' town, California', ''))

	dict_place = {row[1]['place']: row[1].to_dict() for row in df_place.iterrows()}

	for source, node in graph_place._node.items():

		node['population'] = dict_place.get(node['name'], {'population': 0})['population']

	return graph_place

def add_flows():

	pass