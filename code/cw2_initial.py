import numpy as np

def solve(path: str, rounding: bool=True) -> list[list]:
    """
    Your goal is to improve the initial solution for the CVRP problem.

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

    Attention!
      This code works well. You have to improve it!

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

    def compute_savings():
        savings = []

        for i in range(1, n):
            for j in range(i + 1, n):

                s = (
                    distances[0][i]
                    + distances[0][j]
                    - distances[i][j]
                )

                savings.append((s, i, j))

        savings.sort(reverse=True)
        return savings


    def probabilistic_reorder(savings):
        free = {(item[1], item[2]) : item[0] for item in savings}
        new_order = []

        while len(free) > 0:
            tournament_size = random.randint(1, min(20, len(free)))
            tournament = []

            for key in free.keys():
                tournament.append((free[key], key[0], key[1]))
                if len(tournament) == tournament_size:
                    break
            #tournament = random.sample(savings_list, tournament_size)

            total = sum(item[0] for item in tournament)

            if total <= 0:
                item = tournament[-1]
                del free[(item[1], item[2])]
                new_order.append(item)
            else:
                probs = np.cumsum([item[0] for item in tournament]) / total

                prob = random.random()

                for i, item in enumerate(tournament):
                    if prob < probs[i]:
                        del free[(item[1], item[2])]
                        new_order.append(item)
        return new_order

    def build_routes(savings):
        routes = {i: [i] for i in range(1, n)}
        loads = {i: demands[i] for i in range(1, n)}

        cust_route = {i: i for i in range(1, n)}

        for _, i, j in savings:

            ri = cust_route[i]
            rj = cust_route[j]

            if ri == rj:
                continue

            route_i = routes[ri]
            route_j = routes[rj]

            if loads[ri] + loads[rj] > Q:
                continue

            i_start = route_i[0] == i
            i_end = route_i[-1] == i

            j_start = route_j[0] == j
            j_end = route_j[-1] == j

            merged = None

            if i_end and j_start:
                merged = route_i[:] + route_j[:]

            elif i_start and j_end:
                merged = route_j[:] + route_i[:]

            elif i_start and j_start:
                merged = route_i[::-1][:] + route_j[:]

            elif i_end and j_end:
                merged = route_i[:] + route_j[::-1][:]

            if merged is None:
                continue

            new_id = min(ri, rj)

            old_ids = [ri, rj]

            for rid in old_ids:
                if rid in routes:
                    del routes[rid]
                    del loads[rid]

            routes[new_id] = merged
            loads[new_id] = sum(demands[c] for c in merged)

            for c in merged:
                cust_route[c] = new_id

        return list(routes.values())
    
    def route_cost(route):
        cost = 0
        route = [0] + route + [0]
        for i in range(len(route) - 1):
            cost += distances[route[i]][route[i + 1]]
        return cost
    
    def route_load(route):
        return sum(demands[i] for i in route)

    def total_cost(routes):
        return sum(route_cost(r) for r in routes)
    
    def intra_move(route):
        if len(route) <= 1:
            return

        idx = random.randint(0, len(route) - 1)
        customer = route[idx]
        route.pop(idx)

        pos = random.randint(0, len(route) - 1)
        route.insert(pos, customer)

    def inter_move(route1, route2):
        if len(route1) == 0:
            return

        idx = random.randint(0, len(route1) - 1)
        customer = route1[idx]

        if route_load(route2) + demands[customer] > Q:
            return

        route1.pop(idx)

        pos = random.randint(0, max(len(route2) - 1, 0))
        route2.insert(pos, customer)

    def intra_swap(route):
        if len(route) <= 1:
            return

        idxs = random.sample(range(len(route)), k=2)
        route[idxs[0]], route[idxs[1]] = route[idxs[1]], route[idxs[0]]

    def inter_swap(route1, route2):
        if len(route1) == 0 or len(route2) == 0:
            return

        idx1 = random.choice(range(len(route1)))
        idx2 = random.choice(range(len(route2)))

        if route_load(route1) - demands[route1[idx1]] + \
            demands[route2[idx2]] > Q:
            return
        
        if route_load(route2) + demands[route1[idx1]] - \
            demands[route2[idx2]] > Q:
            return
        
        route1[idx1], route2[idx2] = route2[idx2], route1[idx1]

    
    def intra_route(routes):
        for _ in range(100):
            for route in routes:
                for _ in range(50):
                    new_route = route[:]
                    cond = random.randint(1, 2)
                    if cond == 1:
                        intra_move(new_route)
                    else:
                        intra_swap(new_route)

                    cost = route_cost(route)
                    new_cost = route_cost(new_route)

                    if new_cost < cost:
                        route = new_route


    def inter_route(routes):
        for _ in range(100):
            for i in range(len(routes) - 1):
                for j in range(i + 1, len(routes)):

                    for _ in range(50):
                        route1 = routes[i].copy()
                        route2 = routes[j].copy()

                        cond = random.randint(1, 2)
                        if cond == 1:
                            cond2 = random.randint(1, 2)
                            if cond2 == 1:
                                inter_move(route1, route2)
                            else:
                                inter_move(route2, route1)
                        else:
                            inter_swap(route1, route2)

                        cost = route_cost(routes[i]) + route_cost(routes[j])
                        new_cost = route_cost(route1) + route_cost(route2)

                        if new_cost < cost:
                            routes[i] = route1
                            routes[j] = route2

    # -----------------------------------------------------
    # Complete PK algorithm
    # -----------------------------------------------------

    def solve(iterations=500):
        savings = compute_savings()
        best_routes = None
        best_cost = float('inf')
        for _ in range(iterations):
            # Reorder savings probabilistically
            savings_ordered = probabilistic_reorder(savings)
            # Build routes
            routes = build_routes(savings_ordered)
            cost = total_cost(routes)
            if cost < best_cost:
                best_cost = cost
                best_routes = routes


        intra_route(best_routes)
        inter_route(best_routes)

        best_cost = total_cost(best_routes)
        return best_routes, best_cost
    
    routes, _ = solve()
    return routes