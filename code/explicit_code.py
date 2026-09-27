def solve(path: str, rounding: bool=True) -> list[list]:
    """
    Your goal is to find the optimal solution for the CVRP problem.

    You are given `n` points, which represent one 
    depot and n - 1 customers. There are `k` homogeneous
    identical vehicles with capacity `Q` each, which can route between customers and deliver goods. They 
    must start and finish at the depot. The length of the road between two points is the weight of
    the corresponding edge given at the `distances` matrix, clients have their special `demands` of quantity.

    You have to find the most optimal routes for the vehicles, so that they start and end 
    at the depot, every customer is served exactly once and sum of the route demands doesn't
    exceed Q. You can't use more than k vehicles. Try to minimise the total route length.

    Give your answer as a list of routes, where each is a list of numbers 
    containing the sequence of visited customers numbers by a vehicle (!do not include the depots!).

    The length of the resulting list must be equal to the number of the used vehicles.

    input parameters:
      path - path to the task instance in .vrp format
      rounding - whether the distances should be rounded

    output:
      list[list] res - resulting routes

    """
    import numpy as np
    import random
    import vrplib

    path = path.split('.')[0]

    data = vrplib.read_instance(path + '.vrp')
    k = len(vrplib.read_solution(path + '.sol')['routes'])
    n = data['dimension']
    Q = data['capacity']
    distances = data['edge_weight']
    demands = data['demand']
    
    if rounding:
        for ii in range(n):
            for jj in range(n):
                distances[ii][jj] = round(distances[ii][jj])

    pass