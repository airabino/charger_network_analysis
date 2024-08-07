import numpy as np
import pandas as pd
import scipy.stats as st
import networkx as nx

from math import comb
from itertools import combinations
from scipy.stats import f as f_dist
from scipy.stats import rv_histogram

from .utilities import gini

def gravity_from_array(array, weights, min_value = 0):

    weights = np.atleast_2d(weights)

    array[array < min_value] = min_value
    array = 1 / (array  ** 2)
    array[array == np.inf] = min_value
    
    np.fill_diagonal(array, 0)

    return weights @ array @ weights.T

def gini_from_array(array, weights, min_value = 0):

    array *= weights

    array[array < min_value] = min_value
    # array = 1 / (array  ** 2)
    array[array == np.inf] = min_value
    
    np.fill_diagonal(array, 0)

    # return weights @ array

    return gini(array.sum(axis = 1))

def impedance_from_array(array, weights, min_value = 0):


    weights = np.atleast_2d(weights)

    array[array < min_value] = min_value
    array[array == np.inf] = min_value

    np.fill_diagonal(array, 0)

    return weights @ array @ weights.T

def values_to_array(nodes, values, fields):

    node_to_idx = {k: idx for idx, k in enumerate(nodes)}
    idx_to_node = {idx: k for idx, k in enumerate(nodes)}

    n = len(nodes)

    out_arrays = {field: np.ones((n, n)) * -1 for field in fields}

    for idx_s, source in enumerate(nodes):
        for idx_t, target in enumerate(nodes):
            for field, array in out_arrays.items():

                array[idx_s][idx_t] = values[source][target][field]

    return out_arrays

def nodes_visited(nodes, paths):

    visits = {k: 0 for k in nodes}

    for data in paths.values():
        for path in data.values():
            for node in path:

                visits[node] += 1

    return visits

def redundancy_in_station(graph):

    n = []

    for source, node in graph._node.items():

        n.append(node.get('n_dcfc', 0))

    x = list(range(1, max([max(n), 5])))

    h = rv_histogram(np.histogram(n, bins = x))

    return x, n, h

def redundancy_between_stations(graph, field = 'distance', cutoff = 10e3):

    graph_c = graph.copy()

    for source, adj in graph_c._adj.items():
    
        new_adj = {}
        
        for target, edge in adj.items():
    
            if edge[field] <= cutoff:
    
                new_adj[target] = edge
    
        graph_c._adj[source] = new_adj

    cliques = list(nx.find_cliques(graph_c))

    n = []
    
    for clique in cliques:
    
        n.append(sum([graph_c._node[c].get('n_dcfc', 0) for c in clique]))

    x = list(range(1, max([max(n), 5])))

    h = rv_histogram(np.histogram(n, bins = x))

    return x, n, h

def interaction_terms(df, columns = None, order = None):

    n = df.shape[0]

    if columns is None:

        columns = list(df.keys())

    if order is None:

        order = len(columns)

    data = {}

    for level in range(1, order + 2):
        for combination in combinations(columns, level):

            name = combination[0] + ''.join([':' + column for column in combination[1:]])

            values = np.ones(n)

            for column in combination:

                values *= df[column]

            data[name] = values

    return pd.DataFrame(data = data)

def residual_sum_of_squares(x, y):

    return ((x - y) ** 2).sum()

def mean_sum_of_squares(x, y):

    return ((y - x.mean()) ** 2).sum()

def total_sum_of_squares(x):

    return ((x - x.mean()) ** 2).sum()

def coefficient_of_determination(x, y):

    return 1 - (residual_sum_of_squares(x, y) / total_sum_of_squares(x))

def adjusted_coefficient_of_determination(x, y, n, p):

    return 1 - (((1 - coefficient_of_determination(x, y)) * (n - 1)) / (n - p - 1))

def anova_tabular(x, y, n, p):

    sse = residual_sum_of_squares(x, y)
    ssm = mean_sum_of_squares(x, y)
    sst = total_sum_of_squares(x)
    dfe = n - p
    dfm = p - 1
    dft = n - 1

    # print(dfe, dfm, dft)

    mse = sse / dfe
    msm = ssm / dfm
    mst = sst / dft
    f = msm / mse
    pf = f_dist.sf(f, dfm, dfe)

    r2 = coefficient_of_determination(x, y)
    ar2 = adjusted_coefficient_of_determination(x, y, n, p)

    out_string = "\\hline R & R-Squared & Adjusted R-Squared & Std. Error \\\\\n"
    out_string += "\\hline {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\\n".format(
        np.sqrt(r2), r2, ar2, (x - y).std() / n)
    # out_string += "\\hline"

    # print(out_string)

    out_string += "\\hline Category & Sum of Squares & DOF & Mean Squares \\\\\n"
    out_string += "\\hline Model & {:.3f} & {:.0f} & {:.3f} \\\\\n".format(ssm, dfm, msm)
    out_string += "\\hline Error & {:.3f} & {:.0f} & {:.3f} \\\\\n".format(sse, dfe, mse)
    out_string += "\\hline Total & {:.3f} & {:.0f} & {:.3f} \\\\\n".format(sst, dft, mst)
    out_string += "\\hline  \\multicolumn{2}{|c|}{$F$} &  "
    out_string += "\\multicolumn{2}{c|}{$P(>F)$}  \\\\\n"
    out_string += "\\hline  \\multicolumn{{2}}{{|c|}}{{{:.3f}}} &  ".format(f)
    out_string += "\\multicolumn{{2}}{{c|}}{{{:.3f}}}  \\\\\n".format(pf)
    out_string += "\\hline"

    # print(out_string)
    return out_string

def model_anova_tabular(model, df_norm, res_column, n, c = 1):

    y_hat = predict(model, df_norm)
    y = df_norm[res_column]
    m = df_norm.shape[0]
    p = sum([comb(n, k) for k in range(n + 1)]) * c

    # print(n, p, c)

    return anova_tabular(y, y_hat, m, p)

def predict(model, df_norm):

    return model.predict(df_norm)

def significant_parameters_tabular(model, alpha = .05, label_substitutions = {}):
    params = model._results.params
    tvalues = model._results.tvalues
    pvalues = model._results.pvalues
    names = np.array(list(dict(model.params).keys()))
    
    for idx in range(len(names)):

        name = names[idx]

        for key, val in label_substitutions.items():

            if key in name:

                names[idx] = name.replace(key, val)
                name = names[idx]

    params = params[pvalues < alpha]
    tvalues = tvalues[pvalues < alpha]
    names = names[pvalues < alpha]
    pvalues = pvalues[pvalues < alpha]

    name_lengths = [len(name) for name in names]

    name_length_order = np.append(0, np.argsort(name_lengths[1:]) + 1)

    params = params[name_length_order]
    tvalues = tvalues[name_length_order]
    names = names[name_length_order]
    pvalues = pvalues[name_length_order]

    out_string = ""
    for i in range(len(names)):

        out_string += "\\hline {{\\small {} }} & {:.3f} & {:.3f} \\\\\n".format(
            names[i], params[i], pvalues[i]
            )
    
    return out_string

def significant_parameters(model, alpha = .05, just = 0, label_substitutions = {}):
    params = model._results.params
    tvalues = model._results.tvalues
    pvalues = model._results.pvalues
    names = np.array(list(dict(model.params).keys()))
    
    for idx in range(len(names)):

        name = names[idx]

        for key, val in label_substitutions.items():

            if key in name:

                names[idx] = name.replace(key, val)
                name = names[idx]

    params = params[pvalues < alpha]
    tvalues = tvalues[pvalues < alpha]
    names = names[pvalues < alpha]
    pvalues = pvalues[pvalues < alpha]

    name_lengths = [len(name) for name in names]

    name_length_order = np.append(0, np.argsort(name_lengths[1:]) + 1)

    params = params[name_length_order]
    tvalues = tvalues[name_length_order]
    names = names[name_length_order]
    pvalues = pvalues[name_length_order]

    out_dict = {}
    for idx in range(len(names)):

        name_parts = names[idx].split(':')

        name = ''

        for part in name_parts:

            name += part.rjust(just) + '\n'

        name = name[:-1]

        out_dict[name] = {
            'coefficient': params[idx],
            'tvalue': tvalues[idx],
            'pvalue': pvalues[idx],
        }
    
    return out_dict

