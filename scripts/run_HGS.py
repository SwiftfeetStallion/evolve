import sys
import os
import numpy as np
import vrplib
import csv
import subprocess
import time

BASE_PATH = os.getcwd()
sys.path.append(f'{BASE_PATH}/algorithms')

DATA_PATH = f'{BASE_PATH}/data/A' # path to data folder
OUTPUT_PATH = f'{BASE_PATH}/results/algorithms/out.csv'  # save results

RESULT_PATH = f'{BASE_PATH}/results/HGS/output.sol' # save the output of the algorithm

TRIES = 1 # how many times the algorithm should run
SEEDS = np.linspace(10, 101, TRIES)


def run(data_paths, output_path=None, round_int=False):
    stats = []

    for path in data_paths:
        if path.endswith('sol'):
             continue

        data = vrplib.read_instance(path)
        sol = vrplib.read_solution(path.split('.')[0] + '.sol')
        best_cost = sol['cost']
        k = len(sol['routes'])
        
        name = path.split('/')[-1][:-4]


        times = []
        costs = []

        for i in range(TRIES):
            start = time.time()

            command = [
                'python',
                f'{BASE_PATH}/algorithms/hgs_cvrp.py',
                f'{path}',
                RESULT_PATH,
                '-t', f"{int(240 * data['dimension'] / 100)}",
                '-round', f'{int(round_int)}',
                '-veh', f'{k}',
                '-seed', f'{int(SEEDS[i])}',
                '-log', '1'
            ]
            subprocess.run(command)
            res = vrplib.read_solution(RESULT_PATH)
            times.append(round(time.time() - start, 2))
            costs.append(res['cost'])

        gap = max(0, (np.mean(costs).item() - best_cost) * 100 / best_cost)

        stats.append({"name": name,
                      "mean": round(np.mean(costs).item(), 2),
                      "max": max(costs),
                      "min": min(costs),
                      "vehicles": len(res['routes']),
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
