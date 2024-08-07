import os
import re
import sys
import time
import numpy as np
import pandas as pd
import pickle as pkl

import src

directory = '/media/aaron/data/Random_Experiment/'

files = os.listdir(directory)

places = src.graph.graph_from_json('Outputs/places.json')

pop_adj = sum([v['population'] for k, v in places._node.items()]) / len(places)

weighted = {k: v['population'] / 1e6 for k, v in places._node.items()}

relative = {k: v['population'] / pop_adj for k, v in places._node.items()}

unweighted = {k: 1 for k, v in places._node.items()}

functions = {
    'capacity': lambda x: x[1]['capacity'] / 3.6e6,
    'power': lambda x: x[1]['power'] / 1e3,
    'traffic': lambda x: x[2]['station']['traffic'],
    'risk_attitude': lambda x: (x[1]['risk_attitude'][0] + x[1]['risk_attitude'][1]) / 2,
    'reliability': lambda x: x[2]['station']['reliability'],
    'graph_index': lambda x: x[0],
    'gravity': lambda x: src.routing.gravity(
        x[4],
        origins = relative,
        destinations = relative,
        field = 'total_time',
        adjustment = 3600,
    ),
}

paths = {}

outputs = {k: [] for k in functions.keys()}

idx = -1

success = []

for file in src.progress_bar.ProgressBar(files):

    idx += 1

    run = eval(re.findall(r'\d+', file)[0])

    try:

        with open(directory + file, 'rb') as f:
    
            out = pkl.load(f)
    
            # paths[idx] = {'sng': out[0], 'paths': out[5]}
    
        keep = True
        
        for key, fun in functions.items():
    
            try:
    
                outputs[key].append(fun(out))
    
            except:
    
                outputs[key].append(0)
    
                keep = False
    
        if keep:
            
            success.append(idx)
            
    except:
        pass

pkl.dump([outputs, success, paths], open('save_outputs.pkl', 'wb')) 