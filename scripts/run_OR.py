import sys
import os
import vrplib
import numpy as np
import time
import csv

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

BASE_PATH = os.getcwd()
sys.path.append(f'{BASE_PATH}/algorithms')

DATA_PATH = f'{BASE_PATH}/data/A' # path to data folder
OUTPUT_PATH = f'{BASE_PATH}/results/algorithms/out.csv'  # save results
TRIES = 1 # how many times the algorithm should run

def run(data_paths, output_path=None, round_int=True):
    stats = []

    for path in data_paths:
        if path.endswith('sol'):
             continue

        sol = vrplib.read_solution(path.split('.')[0] + '.sol')
        best_cost = sol['cost']
        k = len(sol['routes'])

        data = vrplib.read_instance(path)
        name = path.split('/')[-1][:-4]


        weights = [[0 for _ in range(data['dimension'])] for _ in range(data['dimension'])]
        for i in range(len(data['edge_weight'])):
            for j in range(len(data['edge_weight'])):
                if round_int:
                    weights[i][j] = round(data['edge_weight'][i, j])
                else:
                    weights[i][j] = data['edge_weight'][i, j]

        data['edge_weight'] = weights
        data['demand'] = [int(x) for x in data['demand']]
        data['capacity'] = int(data['capacity'])


        manager = pywrapcp.RoutingIndexManager(
            len(data['edge_weight']), k, 0
        )
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["edge_weight"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return data["demand"][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            [data['capacity']] * k,  # vehicle maximum capacities
            True,  # start cumul to zero
            "Capacity",
        )

        def get_length(manager, routing, solution):
            visited = []
            if solution:
                print(f"Objective: {solution.ObjectiveValue()}")

            for vehicle_id in range(k):
                index = routing.Start(vehicle_id)
                route_load = 0
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    route_load += data["demand"][node_index]
                    index = solution.Value(routing.NextVar(index))

                    if node_index != 0:
                        visited.append(node_index)


                if route_load > data['capacity']:
                    return 1e8

            if len(visited) != data['dimension'] - 1 or len(set(visited)) != len(visited):
                return 1e8

            return solution.ObjectiveValue()

        
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.FromSeconds(int(240 * data['dimension'] / 100))

        def run():
            start = time.time()
            solution = routing.SolveWithParameters(search_parameters)

            if solution:
                cost = get_length(manager, routing, solution)
            else:
                cost = 1e8

            return {
                'time': time.time() - start,
                'cost': cost
            }
        


        times = []
        costs = []

        for _ in range(TRIES):
            res = run()
            times.append(res['time'])
            costs.append(res['cost'])

        gap = max(0, (np.mean(costs).item() - best_cost) * 100 / best_cost)

        stats.append({"name": name,
                      "mean": round(np.mean(costs).item(), 2),
                      "max": max(costs),
                      "min": min(costs),
                      "vehicles": k,
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
