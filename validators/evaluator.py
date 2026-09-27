def test(program_path):
    import vrplib
    import numpy as np
    import importlib.util
    import os

    FAIL = 10000000000
    TRIES = 1

    spec = importlib.util.spec_from_file_location("solution", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def run_test(path, rounding=True):
        path = path.split('.')[0]

        data = vrplib.read_instance(path + '.vrp')
        sol = vrplib.read_solution(path + '.sol')
        edges = data['edge_weight']
        n = data['dimension']
        Q = data['capacity']
        demands = data['demand']

        cost = sol['cost']
        k = len(sol['routes'])

        if rounding:
            for ii in range(n):
                for jj in range(n):
                    edges[ii][jj] = round(edges[ii][jj])

        res = module.solve(path + '.vrp', rounding)

        if not isinstance(res, list):
            raise Exception("The answer is not a list")

        if len(res) > k:
            raise Exception("The number of vehicles is exceeded")

        visited = set([0])
        total = 0

        for i in range(len(res)):
            cap = 0
            route = res[i]
            
            if not isinstance(route, list):
                    raise Exception(f"res[{i}] is not a list")
            
            route = [0] + route + [0]

            for j in range(len(route)):

                if route[j] > 0 and route[j] in visited:
                    raise Exception(f"Client {route[j]} visited twice")
                cap += demands[route[j]]
                visited.add(route[j])

            if cap > Q:
                raise Exception("Route capacity is exceeded")

            total += np.sum([edges[route[j]][[route[j + 1]]] for j in range(len(route) - 1)])

        if len(visited) != n:
            raise Exception("Some clients are not visited")
        
        return total, cost
    
    paths = ['D:/evolve/data/A/A-n32-k5.vrp']
    
    res = {}
    scores = []
    gap = []

    for i, path in enumerate(paths):
        mean_score = 0
        mean_gap = 0
        for _ in range(TRIES):
            try:
                score, cost = run_test(path, rounding=True)
                mean_score += score
                mean_gap += (score - cost) / cost * 100

            except:
                mean_score = FAIL
                mean_gap = FAIL
                break

        mean_gap /= TRIES
        mean_score /= TRIES
        gap.append(mean_gap)

        res[f'score_{i}'] = -mean_score
        scores.append(mean_score)
    res['gap'] = -mean_gap
    res['combined_score'] = 1000 - np.mean(gap) * 0.7 - np.mean(scores) * 0.3
    return res