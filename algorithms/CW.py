from itertools import combinations
from pathlib import Path
import json
import time

class CW:
    """ Clarke-Wright savings heuristic. """
    def __init__(self,
                 n: int, 
                 Q: int, 
                 demands: list[float], 
                 distances: list[list[float]]):
        self.n = n
        self.Q = Q
        self.distances = distances
        self.demands = demands

    def solution_cost(self, sol):
        total_cost = 0
        for item in sol:
            route = [0] + item + [0]
            for k in range(len(route) - 1):
                total_cost += self.distances[route[k]][route[k + 1]]
        return total_cost

    def run(self):
        n = self.n + 1
        routes = {i: [i] for i in range(1, n)}
        route_load = {i: self.demands[i] for i in range(1, n)}

        customer_route = {i: i for i in range(1, n)}

        start = time.time()

        savings = []
        for i, j in combinations(range(1, n), 2):
            s = self.distances[0][i] + self.distances[0][j] - self.distances[i][j]
            savings.append((s, i, j))

        savings.sort(reverse=True)

        for _, (saving, i, j) in enumerate(savings):
    
            ri = customer_route[i]
            rj = customer_route[j]

            if ri == rj:
                continue

            route_i = routes[ri]
            route_j = routes[rj]

            load_i = route_load[ri]
            load_j = route_load[rj]

            if load_i + load_j > self.Q:
                continue

            i_is_start = route_i[0] == i
            i_is_end = route_i[-1] == i

            j_is_start = route_j[0] == j
            j_is_end = route_j[-1] == j

            merged = None

            if i_is_end and j_is_start:
                merged = route_i[:] + route_j[:]

            elif i_is_start and j_is_end:
                merged = route_j[:] + route_i[:]

            elif i_is_start and j_is_start:
                merged = route_i[::-1][:] + route_j[:]

            elif i_is_end and j_is_end:
                merged = route_i[:] + route_j[::-1][:]

            if merged is None:
                continue

            new_id = min(ri, rj)

            routes[new_id] = merged
            route_load[new_id] = load_i + load_j

            for customer in merged:
                if customer != 0:
                    customer_route[customer] = new_id

            del routes[ri]
            del routes[rj]
            del route_load[ri]
            del route_load[rj]

            routes[new_id] = merged
            route_load[new_id] = load_i + load_j



        final_routes = list(routes.values())
        total_cost = self.solution_cost(final_routes)


        result = {
            "cost": total_cost,
            "time": round(time.time() - start, 2),
            "vehicles": len(final_routes)
        }

        return result