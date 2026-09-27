import sys
import os
import vrplib
import numpy as np
import csv

BASE_PATH = os.getcwd()
sys.path.append(f'{BASE_PATH}/algorithms')

DATA_PATH = f'{BASE_PATH}/data/A' # path to data folder
OUTPUT_PATH = f'{BASE_PATH}/results/algorithms/out.csv'  # save results
TRIES = 1 # how many times the algorithm should run

from CW import CW

def run(data_paths, output_path=None, round_int=False):
    stats = []

    for path in data_paths:
        if path.endswith('sol'):
             continue

        sol = vrplib.read_solution(path.split('.')[0] + '.sol')
        best_cost = sol['cost']
        k = len(sol['routes'])


        data = vrplib.read_instance(path)

        if 'distance' in data.keys():
            continue
        n = data['dimension']
        Q = data['capacity']
        #coords = data['node_coord']
        demands = data['demand']
        edges = data['edge_weight']

        if round_int:
            edges = np.round(edges)


        algo = CW(n=n - 1, 
                  Q=Q, 
                  demands=demands, 
                  distances=edges)
        
        name = path.split('/')[-1][:-4]


        times = []
        costs = []

        for _ in range(TRIES):
            res = algo.run()
            times.append(res['time'])
            costs.append(res['cost'])

        gap = max(0, (np.mean(costs).item() - best_cost) / best_cost * 100)

        stats.append({"name": name,
                      "mean": round(np.mean(costs).item(), 2),
                      "max": round(max(costs).item(), 2),
                      "min": round(min(costs).item(), 2),
                      "vehicles": res['vehicles'],
                      "k": k,
                      "BKS": best_cost,
                      "gap": round(gap, 2),
                      "time": round(np.mean(times).item(), 2)})
        

    if output_path:
        with open(output_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=stats[0].keys())
            writer.writeheader()
            writer.writerows(stats)





paths = os.listdir(DATA_PATH)
paths = [DATA_PATH + '/' + item for item in paths]


run(paths, output_path=OUTPUT_PATH, round_int=True)
