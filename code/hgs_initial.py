"""
Python port of HGS-CVRP (Hybrid Genetic Search for Capacitated Vehicle Routing Problem)
Original C++ code by Thibaut Vidal (https://github.com/vidalt/HGS-CVRP)
Python port preserves the structure and algorithm of the original.
"""
# EVOLVE-BLOCK-START

import math
import random
import time
import sys
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# --- Constants ---
MY_EPSILON = 0.00001
PI = 3.14159265359


def clock():
    """Return elapsed CPU time in seconds (like C++ clock()/CLOCKS_PER_SEC)."""
    return time.perf_counter()


# ================================================================
# AlgorithmParameters
# ================================================================
@dataclass
class AlgorithmParameters:
    nbGranular: int = 20
    mu: int = 25
    lambda_: int = 40  # 'lambda' is a keyword in Python
    nbElite: int = 4
    nbClose: int = 5
    nbIterPenaltyManagement: int = 100
    targetFeasible: float = 0.2
    penaltyDecrease: float = 0.85
    penaltyIncrease: float = 1.2
    seed: int = 0
    nbIter: int = 20000
    nbIterTraces: int = 500
    timeLimit: float = 0.0
    useSwapStar: int = 1

    def print_parameters(self):
        print("=========== Algorithm Parameters =================")
        print(f"---- nbGranular              is set to {self.nbGranular}")
        print(f"---- mu                      is set to {self.mu}")
        print(f"---- lambda                  is set to {self.lambda_}")
        print(f"---- nbElite                 is set to {self.nbElite}")
        print(f"---- nbClose                 is set to {self.nbClose}")
        print(f"---- nbIterPenaltyManagement is set to {self.nbIterPenaltyManagement}")
        print(f"---- targetFeasible          is set to {self.targetFeasible}")
        print(f"---- penaltyDecrease         is set to {self.penaltyDecrease}")
        print(f"---- penaltyIncrease         is set to {self.penaltyIncrease}")
        print(f"---- seed                    is set to {self.seed}")
        print(f"---- nbIter                  is set to {self.nbIter}")
        print(f"---- nbIterTraces            is set to {self.nbIterTraces}")
        print(f"---- timeLimit               is set to {self.timeLimit}")
        print(f"---- useSwapStar             is set to {self.useSwapStar}")
        print("==================================================")


def default_algorithm_parameters() -> AlgorithmParameters:
    return AlgorithmParameters()


# ================================================================
# CircleSector
# ================================================================
class CircleSector:
    def __init__(self):
        self.start = 0
        self.end = 0

    @staticmethod
    def positive_mod(i: int) -> int:
        return (i % 65536 + 65536) % 65536

    def initialize(self, point: int):
        self.start = point
        self.end = point

    def is_enclosed(self, point: int) -> bool:
        return CircleSector.positive_mod(point - self.start) <= CircleSector.positive_mod(self.end - self.start)

    @staticmethod
    def overlap(sector1, sector2) -> bool:
        return (CircleSector.positive_mod(sector2.start - sector1.start) <= CircleSector.positive_mod(sector1.end - sector1.start)) or \
               (CircleSector.positive_mod(sector1.start - sector2.start) <= CircleSector.positive_mod(sector2.end - sector2.start))

    def extend(self, point: int):
        if not self.is_enclosed(point):
            if CircleSector.positive_mod(point - self.end) <= CircleSector.positive_mod(self.start - point):
                self.end = point
            else:
                self.start = point


# ================================================================
# Client (data holder)
# ================================================================
@dataclass
class Client:
    coordX: float = 0.0
    coordY: float = 0.0
    serviceDuration: float = 0.0
    demand: float = 0.0
    polarAngle: int = 0


# ================================================================
# Params
# ================================================================
class Params:
    def __init__(self, x_coords: List[float], y_coords: List[float], dist_mtx: List[List[float]],
                 service_time: List[float], demands: List[float],
                 vehicleCapacity: float, durationLimit: float, nbVeh: int,
                 isDurationConstraint: bool, verbose: bool, ap: AlgorithmParameters):
        self.ap = ap
        self.verbose = verbose
        self.isDurationConstraint = isDurationConstraint
        self.nbVehicles = nbVeh
        self.durationLimit = durationLimit
        self.vehicleCapacity = vehicleCapacity
        self.timeCost = dist_mtx  # reference shared in C++, just store it

        self.startTime = clock()
        self.nbClients = len(demands) - 1  # subtract depot
        self.totalDemand = 0.0
        self.maxDemand = 0.0

        # Initialize RNG
        self.ran = random.Random(ap.seed)

        # Check coordinates
        self.areCoordinatesProvided = (len(demands) == len(x_coords)) and (len(demands) == len(y_coords))

        self.cli = [Client() for _ in range(self.nbClients + 1)]
        for i in range(self.nbClients + 1):
            if ap.useSwapStar == 1 and self.areCoordinatesProvided:
                self.cli[i].coordX = x_coords[i]
                self.cli[i].coordY = y_coords[i]
                angle = math.atan2(self.cli[i].coordY - self.cli[0].coordY,
                                   self.cli[i].coordX - self.cli[0].coordX)
                self.cli[i].polarAngle = CircleSector.positive_mod(int(32768. * angle / PI))
            else:
                self.cli[i].coordX = 0.0
                self.cli[i].coordY = 0.0
                self.cli[i].polarAngle = 0

            self.cli[i].serviceDuration = service_time[i]
            self.cli[i].demand = demands[i]
            if self.cli[i].demand > self.maxDemand:
                self.maxDemand = self.cli[i].demand
            self.totalDemand += self.cli[i].demand

        if verbose and ap.useSwapStar == 1 and not self.areCoordinatesProvided:
            print("----- NO COORDINATES HAVE BEEN PROVIDED, SWAP* NEIGHBORHOOD WILL BE DEACTIVATED BY DEFAULT")

        INT_MAX = 10**9
        if nbVeh == INT_MAX:
            self.nbVehicles = int(math.ceil(1.3 * self.totalDemand / vehicleCapacity)) + 3
            if verbose:
                print(f"----- FLEET SIZE WAS NOT SPECIFIED: DEFAULT INITIALIZATION TO {self.nbVehicles} VEHICLES")
        else:
            if verbose:
                print(f"----- FLEET SIZE SPECIFIED: SET TO {self.nbVehicles} VEHICLES")

        self.maxDist = 0.0
        for i in range(self.nbClients + 1):
            for j in range(self.nbClients + 1):
                if self.timeCost[i][j] > self.maxDist:
                    self.maxDist = self.timeCost[i][j]

        # Correlated vertices (granular restriction)
        self.correlatedVertices = [[] for _ in range(self.nbClients + 1)]
        set_corr = [set() for _ in range(self.nbClients + 1)]
        for i in range(1, self.nbClients + 1):
            order_proximity = [(self.timeCost[i][j], j) for j in range(1, self.nbClients + 1) if i != j]
            order_proximity.sort()
            for j in range(min(ap.nbGranular, self.nbClients - 1)):
                set_corr[i].add(order_proximity[j][1])
                set_corr[order_proximity[j][1]].add(i)
        for i in range(1, self.nbClients + 1):
            self.correlatedVertices[i] = list(set_corr[i])

        if self.maxDist < 0.1 or self.maxDist > 100000:
            raise RuntimeError("The distances are of very small or large scale. This could impact numerical stability.")
        if self.maxDemand < 0.1 or self.maxDemand > 100000:
            raise RuntimeError("The demand quantities are of very small or large scale.")
        if self.nbVehicles < math.ceil(self.totalDemand / vehicleCapacity):
            raise RuntimeError("Fleet size is insufficient to service the considered clients.")

        self.penaltyDuration = 1.0
        self.penaltyCapacity = max(0.1, min(1000.0, self.maxDist / self.maxDemand))

        if verbose:
            print(f"----- INSTANCE SUCCESSFULLY LOADED WITH {self.nbClients} CLIENTS AND {self.nbVehicles} VEHICLES")


# ================================================================
# EvalIndiv
# ================================================================
@dataclass
class EvalIndiv:
    penalizedCost: float = 0.0
    nbRoutes: int = 0
    distance: float = 0.0
    capacityExcess: float = 0.0
    durationExcess: float = 0.0
    isFeasible: bool = False


# ================================================================
# Individual
# ================================================================
class Individual:
    def __init__(self, params: Params, fileName: Optional[str] = None):
        self.eval = EvalIndiv()
        self.successors = [0] * (params.nbClients + 1)
        self.predecessors = [0] * (params.nbClients + 1)
        self.chromR = [[] for _ in range(params.nbVehicles)]
        self.chromT = list(range(1, params.nbClients + 1))
        self.indivsPerProximity = {}  # maps distance -> Individual
        self.biasedFitness = 0.0

        if fileName is not None:
            # Read from solution file
            if not os.path.isfile(fileName):
                raise RuntimeError(f"Impossible to open solution file: {fileName}")
            self.chromT = []
            # Read the solution file
            self._load_from_file(fileName, params)
        else:
            # Random individual
            params.ran.shuffle(self.chromT)
            self.eval.penalizedCost = 1e30

    def _load_from_file(self, fileName: str, params: Params):
        with open(fileName, 'r') as f:
            lines = f.readlines()

        self.chromR = [[] for _ in range(params.nbVehicles)]
        route_idx = 0
        i = 0
        while i < len(lines) and route_idx < params.nbVehicles:
            line = lines[i].strip()
            if line.startswith("Route"):
                colon_idx = line.find(':')
                if colon_idx >= 0:
                    nums = line[colon_idx+1:].strip().split()
                    for token in nums:
                        cust = int(token)
                        self.chromT.append(cust)
                        self.chromR[route_idx].append(cust)
                route_idx += 1
            elif line.startswith("Cost"):
                parts = line.split()
                if len(parts) >= 2:
                    read_cost = float(parts[1])
                else:
                    read_cost = 0.0
            i += 1

        self.evaluateCompleteCost(params)
        if len(self.chromT) != params.nbClients:
            raise RuntimeError("Input solution does not contain the correct number of clients")
        if not self.eval.isFeasible:
            raise RuntimeError("Input solution is infeasible")
        if abs(self.eval.penalizedCost - read_cost) > MY_EPSILON:
            raise RuntimeError(f"Input solution has a different cost than announced in the file: {self.eval.penalizedCost} vs {read_cost}")
        if params.verbose:
            print(f"----- INPUT SOLUTION HAS BEEN SUCCESSFULLY READ WITH COST {self.eval.penalizedCost}")

    def evaluateCompleteCost(self, params: Params):
        self.eval = EvalIndiv()
        for r in range(params.nbVehicles):
            if self.chromR[r]:
                distance = params.timeCost[0][self.chromR[r][0]]
                load = params.cli[self.chromR[r][0]].demand
                service = params.cli[self.chromR[r][0]].serviceDuration
                self.predecessors[self.chromR[r][0]] = 0
                for i in range(1, len(self.chromR[r])):
                    c_prev = self.chromR[r][i-1]
                    c_curr = self.chromR[r][i]
                    distance += params.timeCost[c_prev][c_curr]
                    load += params.cli[c_curr].demand
                    service += params.cli[c_curr].serviceDuration
                    self.predecessors[c_curr] = c_prev
                    self.successors[c_prev] = c_curr
                last = self.chromR[r][-1]
                self.successors[last] = 0
                distance += params.timeCost[last][0]
                self.eval.distance += distance
                self.eval.nbRoutes += 1
                if load > params.vehicleCapacity:
                    self.eval.capacityExcess += load - params.vehicleCapacity
                if distance + service > params.durationLimit:
                    self.eval.durationExcess += distance + service - params.durationLimit
        self.eval.penalizedCost = self.eval.distance + self.eval.capacityExcess * params.penaltyCapacity + self.eval.durationExcess * params.penaltyDuration
        self.eval.isFeasible = (self.eval.capacityExcess < MY_EPSILON and self.eval.durationExcess < MY_EPSILON)


# ================================================================
# ClientSplit
# ================================================================
@dataclass
class ClientSplit:
    demand: float = 0.0
    serviceTime: float = 0.0
    d0_x: float = 0.0
    dx_0: float = 0.0
    dnext: float = 0.0


# ================================================================
# Trivial_Deque
# ================================================================
class Trivial_Deque:
    def __init__(self, nbElements: int, firstNode: int):
        self.myDeque = [0] * nbElements
        self.myDeque[0] = firstNode
        self.indexBack = 0
        self.indexFront = 0

    def pop_front(self):
        self.indexFront += 1

    def pop_back(self):
        self.indexBack -= 1

    def push_back(self, i: int):
        self.indexBack += 1
        self.myDeque[self.indexBack] = i

    def get_front(self) -> int:
        return self.myDeque[self.indexFront]

    def get_next_front(self) -> int:
        return self.myDeque[self.indexFront + 1]

    def get_back(self) -> int:
        return self.myDeque[self.indexBack]

    def reset(self, firstNode: int):
        self.myDeque[0] = firstNode
        self.indexBack = 0
        self.indexFront = 0

    def size(self) -> int:
        return self.indexBack - self.indexFront + 1


# ================================================================
# Split
# ================================================================
class Split:
    def __init__(self, params: Params):
        self.params = params
        self.maxVehicles = 0
        self.cliSplit = [ClientSplit() for _ in range(params.nbClients + 1)]
        self.sumDistance = [0.0] * (params.nbClients + 1)
        self.sumLoad = [0.0] * (params.nbClients + 1)
        self.sumService = [0.0] * (params.nbClients + 1)
        self.potential = [[1e30] * (params.nbClients + 1) for _ in range(params.nbVehicles + 1)]
        self.pred = [[0] * (params.nbClients + 1) for _ in range(params.nbVehicles + 1)]

    def propagate(self, i: int, j: int, k: int) -> float:
        return (self.potential[k][i] + self.sumDistance[j] - self.sumDistance[i + 1]
                + self.cliSplit[i + 1].d0_x + self.cliSplit[j].dx_0
                + self.params.penaltyCapacity * max(self.sumLoad[j] - self.sumLoad[i] - self.params.vehicleCapacity, 0.0))

    def dominates(self, i: int, j: int, k: int) -> bool:
        return (self.potential[k][j] + self.cliSplit[j + 1].d0_x >
                self.potential[k][i] + self.cliSplit[i + 1].d0_x + self.sumDistance[j + 1] - self.sumDistance[i + 1]
                + self.params.penaltyCapacity * (self.sumLoad[j] - self.sumLoad[i]))

    def dominatesRight(self, i: int, j: int, k: int) -> bool:
        return (self.potential[k][j] + self.cliSplit[j + 1].d0_x <
                self.potential[k][i] + self.cliSplit[i + 1].d0_x + self.sumDistance[j + 1] - self.sumDistance[i + 1]
                + MY_EPSILON)

    def splitSimple(self, indiv: Individual) -> int:
        self.potential[0][0] = 0
        for i in range(1, self.params.nbClients + 1):
            self.potential[0][i] = 1e30

        if self.params.isDurationConstraint:
            for i in range(self.params.nbClients):
                load = 0.0
                distance = 0.0
                serviceDuration = 0.0
                j = i + 1
                while j <= self.params.nbClients and load <= 1.5 * self.params.vehicleCapacity:
                    load += self.cliSplit[j].demand
                    serviceDuration += self.cliSplit[j].serviceTime
                    if j == i + 1:
                        distance += self.cliSplit[j].d0_x
                    else:
                        distance += self.cliSplit[j - 1].dnext
                    cost = (distance + self.cliSplit[j].dx_0
                            + self.params.penaltyCapacity * max(load - self.params.vehicleCapacity, 0.0)
                            + self.params.penaltyDuration * max(distance + self.cliSplit[j].dx_0 + serviceDuration - self.params.durationLimit, 0.0))
                    if self.potential[0][i] + cost < self.potential[0][j]:
                        self.potential[0][j] = self.potential[0][i] + cost
                        self.pred[0][j] = i
                    j += 1
        else:
            queue = Trivial_Deque(self.params.nbClients + 1, 0)
            for i in range(1, self.params.nbClients + 1):
                self.potential[0][i] = self.propagate(queue.get_front(), i, 0)
                self.pred[0][i] = queue.get_front()
                if i < self.params.nbClients:
                    if not self.dominates(queue.get_back(), i, 0):
                        while queue.size() > 0 and self.dominatesRight(queue.get_back(), i, 0):
                            queue.pop_back()
                        queue.push_back(i)
                    while queue.size() > 1 and self.propagate(queue.get_front(), i + 1, 0) > self.propagate(queue.get_next_front(), i + 1, 0) - MY_EPSILON:
                        queue.pop_front()

        if self.potential[0][self.params.nbClients] > 1e29:
            raise RuntimeError("ERROR : no Split solution has been propagated until the last node")

        for k in range(self.params.nbVehicles - 1, self.maxVehicles - 1, -1):
            indiv.chromR[k].clear()

        end = self.params.nbClients
        for k in range(self.maxVehicles - 1, -1, -1):
            indiv.chromR[k].clear()
            begin = self.pred[0][end]
            for ii in range(begin, end):
                indiv.chromR[k].append(indiv.chromT[ii])
            end = begin

        return 1 if end == 0 else 0

    def splitLF(self, indiv: Individual) -> int:
        self.potential[0][0] = 0
        for k in range(self.maxVehicles + 1):
            for i in range(1, self.params.nbClients + 1):
                self.potential[k][i] = 1e30

        if self.params.isDurationConstraint:
            for k in range(self.maxVehicles):
                i = k
                while i < self.params.nbClients and self.potential[k][i] < 1e29:
                    load = 0.0
                    serviceDuration = 0.0
                    distance = 0.0
                    j = i + 1
                    while j <= self.params.nbClients and load <= 1.5 * self.params.vehicleCapacity:
                        load += self.cliSplit[j].demand
                        serviceDuration += self.cliSplit[j].serviceTime
                        if j == i + 1:
                            distance += self.cliSplit[j].d0_x
                        else:
                            distance += self.cliSplit[j - 1].dnext
                        cost = (distance + self.cliSplit[j].dx_0
                                + self.params.penaltyCapacity * max(load - self.params.vehicleCapacity, 0.0)
                                + self.params.penaltyDuration * max(distance + self.cliSplit[j].dx_0 + serviceDuration - self.params.durationLimit, 0.0))
                        if self.potential[k][i] + cost < self.potential[k + 1][j]:
                            self.potential[k + 1][j] = self.potential[k][i] + cost
                            self.pred[k + 1][j] = i
                        j += 1
                    i += 1
        else:
            queue = Trivial_Deque(self.params.nbClients + 1, 0)
            for k in range(self.maxVehicles):
                queue.reset(k)
                i = k + 1
                while i <= self.params.nbClients and queue.size() > 0:
                    self.potential[k + 1][i] = self.propagate(queue.get_front(), i, k)
                    self.pred[k + 1][i] = queue.get_front()
                    if i < self.params.nbClients:
                        if not self.dominates(queue.get_back(), i, k):
                            while queue.size() > 0 and self.dominatesRight(queue.get_back(), i, k):
                                queue.pop_back()
                            queue.push_back(i)
                        while queue.size() > 1 and self.propagate(queue.get_front(), i + 1, k) > self.propagate(queue.get_next_front(), i + 1, k) - MY_EPSILON:
                            queue.pop_front()
                    i += 1

        if self.potential[self.maxVehicles][self.params.nbClients] > 1e29:
            raise RuntimeError("ERROR : no Split solution has been propagated until the last node")

        minCost = self.potential[self.maxVehicles][self.params.nbClients]
        nbRoutes = self.maxVehicles
        for k in range(1, self.maxVehicles):
            if self.potential[k][self.params.nbClients] < minCost:
                minCost = self.potential[k][self.params.nbClients]
                nbRoutes = k

        for k in range(self.params.nbVehicles - 1, nbRoutes - 1, -1):
            indiv.chromR[k].clear()

        end = self.params.nbClients
        for k in range(nbRoutes - 1, -1, -1):
            indiv.chromR[k].clear()
            begin = self.pred[k + 1][end]
            for ii in range(begin, end):
                indiv.chromR[k].append(indiv.chromT[ii])
            end = begin

        return 1 if end == 0 else 0

    def generalSplit(self, indiv: Individual, nbMaxVehicles: int):
        self.maxVehicles = max(nbMaxVehicles, int(math.ceil(self.params.totalDemand / self.params.vehicleCapacity)))
        for i in range(1, self.params.nbClients + 1):
            self.cliSplit[i].demand = self.params.cli[indiv.chromT[i - 1]].demand
            self.cliSplit[i].serviceTime = self.params.cli[indiv.chromT[i - 1]].serviceDuration
            self.cliSplit[i].d0_x = self.params.timeCost[0][indiv.chromT[i - 1]]
            self.cliSplit[i].dx_0 = self.params.timeCost[indiv.chromT[i - 1]][0]
            if i < self.params.nbClients:
                self.cliSplit[i].dnext = self.params.timeCost[indiv.chromT[i - 1]][indiv.chromT[i]]
            else:
                self.cliSplit[i].dnext = -1e30
            self.sumLoad[i] = self.sumLoad[i - 1] + self.cliSplit[i].demand
            self.sumService[i] = self.sumService[i - 1] + self.cliSplit[i].serviceTime
            self.sumDistance[i] = self.sumDistance[i - 1] + self.cliSplit[i - 1].dnext

        if self.splitSimple(indiv) == 0:
            self.splitLF(indiv)

        indiv.evaluateCompleteCost(self.params)


# ================================================================
# LocalSearch structures
# ================================================================
class Node:
    def __init__(self):
        self.isDepot = False
        self.cour = 0
        self.position = 0
        self.whenLastTestedRI = 0
        self.next = None  # Node
        self.prev = None  # Node
        self.route = None  # Route
        self.cumulatedLoad = 0.0
        self.cumulatedTime = 0.0
        self.cumulatedReversalDistance = 0.0
        self.deltaRemoval = 0.0


class Route:
    def __init__(self):
        self.cour = 0
        self.nbCustomers = 0
        self.whenLastModified = 0
        self.whenLastTestedSWAPStar = 0
        self.depot = None  # Node
        self.duration = 0.0
        self.load = 0.0
        self.reversalDistance = 0.0
        self.penalty = 0.0
        self.polarAngleBarycenter = 0.0
        self.sector = CircleSector()


class ThreeBestInsert:
    def __init__(self):
        self.whenLastCalculated = 0
        self.bestCost = [1e30, 1e30, 1e30]
        self.bestLocation = [None, None, None]

    def compareAndAdd(self, costInsert: float, placeInsert: 'Node'):
        if costInsert >= self.bestCost[2]:
            return
        elif costInsert >= self.bestCost[1]:
            self.bestCost[2] = costInsert
            self.bestLocation[2] = placeInsert
        elif costInsert >= self.bestCost[0]:
            self.bestCost[2] = self.bestCost[1]
            self.bestLocation[2] = self.bestLocation[1]
            self.bestCost[1] = costInsert
            self.bestLocation[1] = placeInsert
        else:
            self.bestCost[2] = self.bestCost[1]
            self.bestLocation[2] = self.bestLocation[1]
            self.bestCost[1] = self.bestCost[0]
            self.bestLocation[1] = self.bestLocation[0]
            self.bestCost[0] = costInsert
            self.bestLocation[0] = placeInsert

    def reset(self):
        self.bestCost = [1e30, 1e30, 1e30]
        self.bestLocation = [None, None, None]


class SwapStarElement:
    def __init__(self):
        self.moveCost = 1e30
        self.U = None
        self.bestPositionU = None
        self.V = None
        self.bestPositionV = None


# ================================================================
# LocalSearch
# ================================================================
class LocalSearch:
    def __init__(self, params: Params):
        self.params = params
        self.searchCompleted = False
        self.nbMoves = 0
        self.orderNodes = list(range(1, params.nbClients + 1))
        self.orderRoutes = list(range(params.nbVehicles))
        self.emptyRoutes = set()
        self.loopID = 0

        # Solution representation
        self.clients = [Node() for _ in range(params.nbClients + 1)]
        self.routes = [Route() for _ in range(params.nbVehicles)]
        self.depots = [Node() for _ in range(params.nbVehicles)]
        self.depotsEnd = [Node() for _ in range(params.nbVehicles)]
        self.bestInsertClient = [[ThreeBestInsert() for _ in range(params.nbClients + 1)] for _ in range(params.nbVehicles)]

        # Temp variables for local search
        self.nodeU = None
        self.nodeX = None
        self.nodeV = None
        self.nodeY = None
        self.routeU = None
        self.routeV = None
        self.nodeUPrevIndex = 0
        self.nodeUIndex = 0
        self.nodeXIndex = 0
        self.nodeXNextIndex = 0
        self.nodeVPrevIndex = 0
        self.nodeVIndex = 0
        self.nodeYIndex = 0
        self.nodeYNextIndex = 0
        self.loadU = 0.0
        self.loadX = 0.0
        self.loadV = 0.0
        self.loadY = 0.0
        self.serviceU = 0.0
        self.serviceX = 0.0
        self.serviceV = 0.0
        self.serviceY = 0.0
        self.penaltyCapacityLS = 0.0
        self.penaltyDurationLS = 0.0
        self.intraRouteMove = False

        # Initialize node indices
        for i in range(params.nbClients + 1):
            self.clients[i].cour = i
            self.clients[i].isDepot = False
        for i in range(params.nbVehicles):
            self.routes[i].cour = i
            self.routes[i].depot = self.depots[i]
            self.depots[i].cour = 0
            self.depots[i].isDepot = True
            self.depots[i].route = self.routes[i]
            self.depotsEnd[i].cour = 0
            self.depotsEnd[i].isDepot = True
            self.depotsEnd[i].route = self.routes[i]

    def run(self, indiv: Individual, penaltyCapacityLS: float, penaltyDurationLS: float):
        self.penaltyCapacityLS = penaltyCapacityLS
        self.penaltyDurationLS = penaltyDurationLS
        self.loadIndividual(indiv)

        self.params.ran.shuffle(self.orderNodes)
        self.params.ran.shuffle(self.orderRoutes)
        for i in range(1, self.params.nbClients + 1):
            if self.params.ran.randint(0, self.params.ap.nbGranular - 1) == 0:
                self.params.ran.shuffle(self.params.correlatedVertices[i])

        self.searchCompleted = False
        loopID = 0
        while not self.searchCompleted:
            if loopID > 1:
                self.searchCompleted = True

            # RI moves
            for posU in range(self.params.nbClients):
                self.nodeU = self.clients[self.orderNodes[posU]]
                lastTestRINodeU = self.nodeU.whenLastTestedRI
                self.nodeU.whenLastTestedRI = self.nbMoves

                for posV in range(len(self.params.correlatedVertices[self.nodeU.cour])):
                    nodeV_idx = self.params.correlatedVertices[self.nodeU.cour][posV]
                    self.nodeV = self.clients[nodeV_idx]
                    if loopID == 0 or max(self.nodeU.route.whenLastModified, self.nodeV.route.whenLastModified) > lastTestRINodeU:
                        self.setLocalVariablesRouteU()
                        self.setLocalVariablesRouteV()
                        if self.move1(): continue
                        if self.move2(): continue
                        if self.move3(): continue
                        if self.nodeUIndex <= self.nodeVIndex and self.move4(): continue
                        if self.move5(): continue
                        if self.nodeUIndex <= self.nodeVIndex and self.move6(): continue
                        if self.intraRouteMove and self.move7(): continue
                        if not self.intraRouteMove and self.move8(): continue
                        if not self.intraRouteMove and self.move9(): continue

                        if self.nodeV.prev.isDepot:
                            self.nodeV = self.nodeV.prev
                            self.setLocalVariablesRouteV()
                            if self.move1(): continue
                            if self.move2(): continue
                            if self.move3(): continue
                            if not self.intraRouteMove and self.move8(): continue
                            if not self.intraRouteMove and self.move9(): continue

                # Moves involving empty routes
                if loopID > 0 and self.emptyRoutes:
                    self.nodeV = self.routes[list(self.emptyRoutes)[0]].depot
                    self.setLocalVariablesRouteU()
                    self.setLocalVariablesRouteV()
                    if self.move1(): continue
                    if self.move2(): continue
                    if self.move3(): continue
                    if self.move9(): continue

            # SWAP* moves
            if self.params.ap.useSwapStar == 1 and self.params.areCoordinatesProvided:
                for rU in range(self.params.nbVehicles):
                    self.routeU = self.routes[self.orderRoutes[rU]]
                    lastTestSWAPStarRouteU = self.routeU.whenLastTestedSWAPStar
                    self.routeU.whenLastTestedSWAPStar = self.nbMoves
                    for rV in range(self.params.nbVehicles):
                        self.routeV = self.routes[self.orderRoutes[rV]]
                        if (self.routeU.nbCustomers > 0 and self.routeV.nbCustomers > 0
                                and self.routeU.cour < self.routeV.cour
                                and (loopID == 0 or max(self.routeU.whenLastModified, self.routeV.whenLastModified) > lastTestSWAPStarRouteU)):
                            if CircleSector.overlap(self.routeU.sector, self.routeV.sector):
                                self.swapStar()

            loopID += 1

        self.exportIndividual(indiv)

    def setLocalVariablesRouteU(self):
        self.routeU = self.nodeU.route
        self.nodeX = self.nodeU.next
        self.nodeXNextIndex = self.nodeX.next.cour
        self.nodeUIndex = self.nodeU.cour
        self.nodeUPrevIndex = self.nodeU.prev.cour
        self.nodeXIndex = self.nodeX.cour
        self.loadU = self.params.cli[self.nodeUIndex].demand
        self.serviceU = self.params.cli[self.nodeUIndex].serviceDuration
        self.loadX = self.params.cli[self.nodeXIndex].demand
        self.serviceX = self.params.cli[self.nodeXIndex].serviceDuration

    def setLocalVariablesRouteV(self):
        self.routeV = self.nodeV.route
        self.nodeY = self.nodeV.next
        self.nodeYNextIndex = self.nodeY.next.cour
        self.nodeVIndex = self.nodeV.cour
        self.nodeVPrevIndex = self.nodeV.prev.cour
        self.nodeYIndex = self.nodeY.cour
        self.loadV = self.params.cli[self.nodeVIndex].demand
        self.serviceV = self.params.cli[self.nodeVIndex].serviceDuration
        self.loadY = self.params.cli[self.nodeYIndex].demand
        self.serviceY = self.params.cli[self.nodeYIndex].serviceDuration
        self.intraRouteMove = (self.routeU == self.routeV)

    def penaltyExcessDuration(self, myDuration: float) -> float:
        return max(0.0, myDuration - self.params.durationLimit) * self.penaltyDurationLS

    def penaltyExcessLoad(self, myLoad: float) -> float:
        return max(0.0, myLoad - self.params.vehicleCapacity) * self.penaltyCapacityLS

    def move1(self) -> bool:
        costSuppU = (self.params.timeCost[self.nodeUPrevIndex][self.nodeXIndex]
                     - self.params.timeCost[self.nodeUPrevIndex][self.nodeUIndex]
                     - self.params.timeCost[self.nodeUIndex][self.nodeXIndex])
        costSuppV = (self.params.timeCost[self.nodeVIndex][self.nodeUIndex]
                     + self.params.timeCost[self.nodeUIndex][self.nodeYIndex]
                     - self.params.timeCost[self.nodeVIndex][self.nodeYIndex])
        if not self.intraRouteMove:
            if costSuppU + costSuppV >= self.routeU.penalty + self.routeV.penalty:
                return False
            costSuppU += (self.penaltyExcessDuration(self.routeU.duration + costSuppU - self.serviceU)
                          + self.penaltyExcessLoad(self.routeU.load - self.loadU)
                          - self.routeU.penalty)
            costSuppV += (self.penaltyExcessDuration(self.routeV.duration + costSuppV + self.serviceU)
                          + self.penaltyExcessLoad(self.routeV.load + self.loadU)
                          - self.routeV.penalty)
        if costSuppU + costSuppV > -MY_EPSILON:
            return False
        if self.nodeUIndex == self.nodeYIndex:
            return False
        self.insertNode(self.nodeU, self.nodeV)
        self.nbMoves += 1
        self.searchCompleted = False
        self.updateRouteData(self.routeU)
        if not self.intraRouteMove:
            self.updateRouteData(self.routeV)
        return True

    def move2(self) -> bool:
        costSuppU = (self.params.timeCost[self.nodeUPrevIndex][self.nodeXNextIndex]
                     - self.params.timeCost[self.nodeUPrevIndex][self.nodeUIndex]
                     - self.params.timeCost[self.nodeXIndex][self.nodeXNextIndex])
        costSuppV = (self.params.timeCost[self.nodeVIndex][self.nodeUIndex]
                     + self.params.timeCost[self.nodeXIndex][self.nodeYIndex]
                     - self.params.timeCost[self.nodeVIndex][self.nodeYIndex])
        if not self.intraRouteMove:
            if costSuppU + costSuppV >= self.routeU.penalty + self.routeV.penalty:
                return False
            costSuppU += (self.penaltyExcessDuration(self.routeU.duration + costSuppU - self.params.timeCost[self.nodeUIndex][self.nodeXIndex] - self.serviceU - self.serviceX)
                          + self.penaltyExcessLoad(self.routeU.load - self.loadU - self.loadX)
                          - self.routeU.penalty)
            costSuppV += (self.penaltyExcessDuration(self.routeV.duration + costSuppV + self.params.timeCost[self.nodeUIndex][self.nodeXIndex] + self.serviceU + self.serviceX)
                          + self.penaltyExcessLoad(self.routeV.load + self.loadU + self.loadX)
                          - self.routeV.penalty)
        if costSuppU + costSuppV > -MY_EPSILON:
            return False
        if self.nodeU == self.nodeY or self.nodeV == self.nodeX or self.nodeX.isDepot:
            return False
        self.insertNode(self.nodeU, self.nodeV)
        self.insertNode(self.nodeX, self.nodeU)
        self.nbMoves += 1
        self.searchCompleted = False
        self.updateRouteData(self.routeU)
        if not self.intraRouteMove:
            self.updateRouteData(self.routeV)
        return True

    def move3(self) -> bool:
        costSuppU = (self.params.timeCost[self.nodeUPrevIndex][self.nodeXNextIndex]
                     - self.params.timeCost[self.nodeUPrevIndex][self.nodeUIndex]
                     - self.params.timeCost[self.nodeUIndex][self.nodeXIndex]
                     - self.params.timeCost[self.nodeXIndex][self.nodeXNextIndex])
        costSuppV = (self.params.timeCost[self.nodeVIndex][self.nodeXIndex]
                     + self.params.timeCost[self.nodeXIndex][self.nodeUIndex]
                     + self.params.timeCost[self.nodeUIndex][self.nodeYIndex]
                     - self.params.timeCost[self.nodeVIndex][self.nodeYIndex])
        if not self.intraRouteMove:
            if costSuppU + costSuppV >= self.routeU.penalty + self.routeV.penalty:
                return False
            costSuppU += (self.penaltyExcessDuration(self.routeU.duration + costSuppU - self.serviceU - self.serviceX)
                          + self.penaltyExcessLoad(self.routeU.load - self.loadU - self.loadX)
                          - self.routeU.penalty)
            costSuppV += (self.penaltyExcessDuration(self.routeV.duration + costSuppV + self.serviceU + self.serviceX)
                          + self.penaltyExcessLoad(self.routeV.load + self.loadU + self.loadX)
                          - self.routeV.penalty)
        if costSuppU + costSuppV > -MY_EPSILON:
            return False
        if self.nodeU == self.nodeY or self.nodeX == self.nodeV or self.nodeX.isDepot:
            return False
        self.insertNode(self.nodeX, self.nodeV)
        self.insertNode(self.nodeU, self.nodeX)
        self.nbMoves += 1
        self.searchCompleted = False
        self.updateRouteData(self.routeU)
        if not self.intraRouteMove:
            self.updateRouteData(self.routeV)
        return True

    def move4(self) -> bool:
        costSuppU = (self.params.timeCost[self.nodeUPrevIndex][self.nodeVIndex]
                     + self.params.timeCost[self.nodeVIndex][self.nodeXIndex]
                     - self.params.timeCost[self.nodeUPrevIndex][self.nodeUIndex]
                     - self.params.timeCost[self.nodeUIndex][self.nodeXIndex])
        costSuppV = (self.params.timeCost[self.nodeVPrevIndex][self.nodeUIndex]
                     + self.params.timeCost[self.nodeUIndex][self.nodeYIndex]
                     - self.params.timeCost[self.nodeVPrevIndex][self.nodeVIndex]
                     - self.params.timeCost[self.nodeVIndex][self.nodeYIndex])
        if not self.intraRouteMove:
            if costSuppU + costSuppV >= self.routeU.penalty + self.routeV.penalty:
                return False
            costSuppU += (self.penaltyExcessDuration(self.routeU.duration + costSuppU + self.serviceV - self.serviceU)
                          + self.penaltyExcessLoad(self.routeU.load + self.loadV - self.loadU)
                          - self.routeU.penalty)
            costSuppV += (self.penaltyExcessDuration(self.routeV.duration + costSuppV - self.serviceV + self.serviceU)
                          + self.penaltyExcessLoad(self.routeV.load + self.loadU - self.loadV)
                          - self.routeV.penalty)
        if costSuppU + costSuppV > -MY_EPSILON:
            return False
        if self.nodeUIndex == self.nodeVPrevIndex or self.nodeUIndex == self.nodeYIndex:
            return False
        self.swapNode(self.nodeU, self.nodeV)
        self.nbMoves += 1
        self.searchCompleted = False
        self.updateRouteData(self.routeU)
        if not self.intraRouteMove:
            self.updateRouteData(self.routeV)
        return True

    def move5(self) -> bool:
        costSuppU = (self.params.timeCost[self.nodeUPrevIndex][self.nodeVIndex]
                     + self.params.timeCost[self.nodeVIndex][self.nodeXNextIndex]
                     - self.params.timeCost[self.nodeUPrevIndex][self.nodeUIndex]
                     - self.params.timeCost[self.nodeXIndex][self.nodeXNextIndex])
        costSuppV = (self.params.timeCost[self.nodeVPrevIndex][self.nodeUIndex]
                     + self.params.timeCost[self.nodeXIndex][self.nodeYIndex]
                     - self.params.timeCost[self.nodeVPrevIndex][self.nodeVIndex]
                     - self.params.timeCost[self.nodeVIndex][self.nodeYIndex])
        if not self.intraRouteMove:
            if costSuppU + costSuppV >= self.routeU.penalty + self.routeV.penalty:
                return False
            costSuppU += (self.penaltyExcessDuration(self.routeU.duration + costSuppU - self.params.timeCost[self.nodeUIndex][self.nodeXIndex] + self.serviceV - self.serviceU - self.serviceX)
                          + self.penaltyExcessLoad(self.routeU.load + self.loadV - self.loadU - self.loadX)
                          - self.routeU.penalty)
            costSuppV += (self.penaltyExcessDuration(self.routeV.duration + costSuppV + self.params.timeCost[self.nodeUIndex][self.nodeXIndex] - self.serviceV + self.serviceU + self.serviceX)
                          + self.penaltyExcessLoad(self.routeV.load + self.loadU + self.loadX - self.loadV)
                          - self.routeV.penalty)
        if costSuppU + costSuppV > -MY_EPSILON:
            return False
        if (self.nodeU == self.nodeV.prev or self.nodeX == self.nodeV.prev
                or self.nodeU == self.nodeY or self.nodeX.isDepot):
            return False
        self.swapNode(self.nodeU, self.nodeV)
        self.insertNode(self.nodeX, self.nodeU)
        self.nbMoves += 1
        self.searchCompleted = False
        self.updateRouteData(self.routeU)
        if not self.intraRouteMove:
            self.updateRouteData(self.routeV)
        return True

    def move6(self) -> bool:
        costSuppU = (self.params.timeCost[self.nodeUPrevIndex][self.nodeVIndex]
                     + self.params.timeCost[self.nodeYIndex][self.nodeXNextIndex]
                     - self.params.timeCost[self.nodeUPrevIndex][self.nodeUIndex]
                     - self.params.timeCost[self.nodeXIndex][self.nodeXNextIndex])
        costSuppV = (self.params.timeCost[self.nodeVPrevIndex][self.nodeUIndex]
                     + self.params.timeCost[self.nodeXIndex][self.nodeYNextIndex]
                     - self.params.timeCost[self.nodeVPrevIndex][self.nodeVIndex]
                     - self.params.timeCost[self.nodeYIndex][self.nodeYNextIndex])
        if not self.intraRouteMove:
            if costSuppU + costSuppV >= self.routeU.penalty + self.routeV.penalty:
                return False
            costSuppU += (self.penaltyExcessDuration(self.routeU.duration + costSuppU - self.params.timeCost[self.nodeUIndex][self.nodeXIndex] + self.params.timeCost[self.nodeVIndex][self.nodeYIndex] + self.serviceV + self.serviceY - self.serviceU - self.serviceX)
                          + self.penaltyExcessLoad(self.routeU.load + self.loadV + self.loadY - self.loadU - self.loadX)
                          - self.routeU.penalty)
            costSuppV += (self.penaltyExcessDuration(self.routeV.duration + costSuppV + self.params.timeCost[self.nodeUIndex][self.nodeXIndex] - self.params.timeCost[self.nodeVIndex][self.nodeYIndex] - self.serviceV - self.serviceY + self.serviceU + self.serviceX)
                          + self.penaltyExcessLoad(self.routeV.load + self.loadU + self.loadX - self.loadV - self.loadY)
                          - self.routeV.penalty)
        if costSuppU + costSuppV > -MY_EPSILON:
            return False
        if (self.nodeX.isDepot or self.nodeY.isDepot or self.nodeY == self.nodeU.prev
                or self.nodeU == self.nodeY or self.nodeX == self.nodeV or self.nodeV == self.nodeX.next):
            return False
        self.swapNode(self.nodeU, self.nodeV)
        self.swapNode(self.nodeX, self.nodeY)
        self.nbMoves += 1
        self.searchCompleted = False
        self.updateRouteData(self.routeU)
        if not self.intraRouteMove:
            self.updateRouteData(self.routeV)
        return True

    def move7(self) -> bool:
        if self.nodeU.position > self.nodeV.position:
            return False
        cost = (self.params.timeCost[self.nodeUIndex][self.nodeVIndex]
                + self.params.timeCost[self.nodeXIndex][self.nodeYIndex]
                - self.params.timeCost[self.nodeUIndex][self.nodeXIndex]
                - self.params.timeCost[self.nodeVIndex][self.nodeYIndex]
                + self.nodeV.cumulatedReversalDistance - self.nodeX.cumulatedReversalDistance)
        if cost > -MY_EPSILON:
            return False
        if self.nodeU.next == self.nodeV:
            return False

        nodeNum = self.nodeX.next
        self.nodeX.prev = nodeNum
        self.nodeX.next = self.nodeY

        while nodeNum != self.nodeV:
            temp = nodeNum.next
            nodeNum.next = nodeNum.prev
            nodeNum.prev = temp
            nodeNum = temp
        self.nodeV.next = self.nodeV.prev
        self.nodeV.prev = self.nodeU
        self.nodeU.next = self.nodeV
        self.nodeY.prev = self.nodeX

        self.nbMoves += 1
        self.searchCompleted = False
        self.updateRouteData(self.routeU)
        return True

    def move8(self) -> bool:
        cost = (self.params.timeCost[self.nodeUIndex][self.nodeVIndex]
                + self.params.timeCost[self.nodeXIndex][self.nodeYIndex]
                - self.params.timeCost[self.nodeUIndex][self.nodeXIndex]
                - self.params.timeCost[self.nodeVIndex][self.nodeYIndex]
                + self.nodeV.cumulatedReversalDistance + self.routeU.reversalDistance
                - self.nodeX.cumulatedReversalDistance
                - self.routeU.penalty - self.routeV.penalty)
        if cost >= 0:
            return False

        cost += (self.penaltyExcessDuration(self.nodeU.cumulatedTime + self.nodeV.cumulatedTime + self.nodeV.cumulatedReversalDistance + self.params.timeCost[self.nodeUIndex][self.nodeVIndex])
                 + self.penaltyExcessDuration(self.routeU.duration - self.nodeU.cumulatedTime - self.params.timeCost[self.nodeUIndex][self.nodeXIndex] + self.routeU.reversalDistance - self.nodeX.cumulatedReversalDistance + self.routeV.duration - self.nodeV.cumulatedTime - self.params.timeCost[self.nodeVIndex][self.nodeYIndex] + self.params.timeCost[self.nodeXIndex][self.nodeYIndex])
                 + self.penaltyExcessLoad(self.nodeU.cumulatedLoad + self.nodeV.cumulatedLoad)
                 + self.penaltyExcessLoad(self.routeU.load + self.routeV.load - self.nodeU.cumulatedLoad - self.nodeV.cumulatedLoad))

        if cost > -MY_EPSILON:
            return False

        depotU = self.routeU.depot
        depotV = self.routeV.depot
        depotUFin = self.routeU.depot.prev
        depotVFin = self.routeV.depot.prev
        depotVSuiv = depotV.next

        xx = self.nodeX
        vv = self.nodeV

        while not xx.isDepot:
            temp = xx.next
            xx.next = xx.prev
            xx.prev = temp
            xx.route = self.routeV
            xx = temp

        while not vv.isDepot:
            temp = vv.prev
            vv.prev = vv.next
            vv.next = temp
            vv.route = self.routeU
            vv = temp

        self.nodeU.next = self.nodeV
        self.nodeV.prev = self.nodeU
        self.nodeX.next = self.nodeY
        self.nodeY.prev = self.nodeX

        if self.nodeX.isDepot:
            depotUFin.next = depotU
            depotUFin.prev = depotVSuiv
            depotUFin.prev.next = depotUFin
            depotV.next = self.nodeY
            self.nodeY.prev = depotV
        elif self.nodeV.isDepot:
            depotV.next = depotUFin.prev
            depotV.next.prev = depotV
            depotV.prev = depotVFin
            depotUFin.prev = self.nodeU
            self.nodeU.next = depotUFin
        else:
            depotV.next = depotUFin.prev
            depotV.next.prev = depotV
            depotUFin.prev = depotVSuiv
            depotUFin.prev.next = depotUFin

        self.nbMoves += 1
        self.searchCompleted = False
        self.updateRouteData(self.routeU)
        self.updateRouteData(self.routeV)
        return True

    def move9(self) -> bool:
        cost = (self.params.timeCost[self.nodeUIndex][self.nodeYIndex]
                + self.params.timeCost[self.nodeVIndex][self.nodeXIndex]
                - self.params.timeCost[self.nodeUIndex][self.nodeXIndex]
                - self.params.timeCost[self.nodeVIndex][self.nodeYIndex]
                - self.routeU.penalty - self.routeV.penalty)
        if cost >= 0:
            return False

        cost += (self.penaltyExcessDuration(self.nodeU.cumulatedTime + self.routeV.duration - self.nodeV.cumulatedTime - self.params.timeCost[self.nodeVIndex][self.nodeYIndex] + self.params.timeCost[self.nodeUIndex][self.nodeYIndex])
                 + self.penaltyExcessDuration(self.routeU.duration - self.nodeU.cumulatedTime - self.params.timeCost[self.nodeUIndex][self.nodeXIndex] + self.nodeV.cumulatedTime + self.params.timeCost[self.nodeVIndex][self.nodeXIndex])
                 + self.penaltyExcessLoad(self.nodeU.cumulatedLoad + self.routeV.load - self.nodeV.cumulatedLoad)
                 + self.penaltyExcessLoad(self.nodeV.cumulatedLoad + self.routeU.load - self.nodeU.cumulatedLoad))

        if cost > -MY_EPSILON:
            return False

        depotU = self.routeU.depot
        depotV = self.routeV.depot
        depotUFin = depotU.prev
        depotVFin = depotV.prev
        depotUpred = depotUFin.prev

        count = self.nodeY
        while not count.isDepot:
            count.route = self.routeU
            count = count.next

        count = self.nodeX
        while not count.isDepot:
            count.route = self.routeV
            count = count.next

        self.nodeU.next = self.nodeY
        self.nodeY.prev = self.nodeU
        self.nodeV.next = self.nodeX
        self.nodeX.prev = self.nodeV

        if self.nodeX.isDepot:
            depotUFin.prev = depotVFin.prev
            depotUFin.prev.next = depotUFin
            self.nodeV.next = depotVFin
            depotVFin.prev = self.nodeV
        else:
            depotUFin.prev = depotVFin.prev
            depotUFin.prev.next = depotUFin
            depotVFin.prev = depotUpred
            depotVFin.prev.next = depotVFin

        self.nbMoves += 1
        self.searchCompleted = False
        self.updateRouteData(self.routeU)
        self.updateRouteData(self.routeV)
        return True

    def swapStar(self) -> bool:
        myBestSwapStar = SwapStarElement()

        self.preprocessInsertions(self.routeU, self.routeV)
        self.preprocessInsertions(self.routeV, self.routeU)

        # Evaluate SWAP* moves
        self.nodeU = self.routeU.depot.next
        while not self.nodeU.isDepot:
            self.nodeV = self.routeV.depot.next
            while not self.nodeV.isDepot:
                deltaPenRouteU = (self.penaltyExcessLoad(self.routeU.load + self.params.cli[self.nodeV.cour].demand - self.params.cli[self.nodeU.cour].demand)
                                  - self.routeU.penalty)
                deltaPenRouteV = (self.penaltyExcessLoad(self.routeV.load + self.params.cli[self.nodeU.cour].demand - self.params.cli[self.nodeV.cour].demand)
                                  - self.routeV.penalty)

                if deltaPenRouteU + self.nodeU.deltaRemoval + deltaPenRouteV + self.nodeV.deltaRemoval <= 0:
                    mySwapStar = SwapStarElement()
                    mySwapStar.U = self.nodeU
                    mySwapStar.V = self.nodeV
                    extraV, mySwapStar.bestPositionU = self.getCheapestInsertSimultRemoval(self.nodeU, self.nodeV)
                    extraU, mySwapStar.bestPositionV = self.getCheapestInsertSimultRemoval(self.nodeV, self.nodeU)

                    mySwapStar.moveCost = (deltaPenRouteU + self.nodeU.deltaRemoval + extraU
                                           + deltaPenRouteV + self.nodeV.deltaRemoval + extraV
                                           + self.penaltyExcessDuration(self.routeU.duration + self.nodeU.deltaRemoval + extraU + self.params.cli[self.nodeV.cour].serviceDuration - self.params.cli[self.nodeU.cour].serviceDuration)
                                           + self.penaltyExcessDuration(self.routeV.duration + self.nodeV.deltaRemoval + extraV - self.params.cli[self.nodeV.cour].serviceDuration + self.params.cli[self.nodeU.cour].serviceDuration))

                    if mySwapStar.moveCost < myBestSwapStar.moveCost:
                        myBestSwapStar = mySwapStar
                self.nodeV = self.nodeV.next
            self.nodeU = self.nodeU.next

        # RELOCATE from routeU to routeV
        self.nodeU = self.routeU.depot.next
        while not self.nodeU.isDepot:
            mySwapStar = SwapStarElement()
            mySwapStar.U = self.nodeU
            mySwapStar.bestPositionU = self.bestInsertClient[self.routeV.cour][self.nodeU.cour].bestLocation[0]
            deltaDistRouteU = (self.params.timeCost[self.nodeU.prev.cour][self.nodeU.next.cour]
                               - self.params.timeCost[self.nodeU.prev.cour][self.nodeU.cour]
                               - self.params.timeCost[self.nodeU.cour][self.nodeU.next.cour])
            deltaDistRouteV = self.bestInsertClient[self.routeV.cour][self.nodeU.cour].bestCost[0]
            mySwapStar.moveCost = (deltaDistRouteU + deltaDistRouteV
                                   + self.penaltyExcessLoad(self.routeU.load - self.params.cli[self.nodeU.cour].demand) - self.routeU.penalty
                                   + self.penaltyExcessLoad(self.routeV.load + self.params.cli[self.nodeU.cour].demand) - self.routeV.penalty
                                   + self.penaltyExcessDuration(self.routeU.duration + deltaDistRouteU - self.params.cli[self.nodeU.cour].serviceDuration)
                                   + self.penaltyExcessDuration(self.routeV.duration + deltaDistRouteV + self.params.cli[self.nodeU.cour].serviceDuration))
            if mySwapStar.moveCost < myBestSwapStar.moveCost:
                myBestSwapStar = mySwapStar
            self.nodeU = self.nodeU.next

        # RELOCATE from routeV to routeU
        self.nodeV = self.routeV.depot.next
        while not self.nodeV.isDepot:
            mySwapStar = SwapStarElement()
            mySwapStar.V = self.nodeV
            mySwapStar.bestPositionV = self.bestInsertClient[self.routeU.cour][self.nodeV.cour].bestLocation[0]
            deltaDistRouteU = self.bestInsertClient[self.routeU.cour][self.nodeV.cour].bestCost[0]
            deltaDistRouteV = (self.params.timeCost[self.nodeV.prev.cour][self.nodeV.next.cour]
                               - self.params.timeCost[self.nodeV.prev.cour][self.nodeV.cour]
                               - self.params.timeCost[self.nodeV.cour][self.nodeV.next.cour])
            mySwapStar.moveCost = (deltaDistRouteU + deltaDistRouteV
                                   + self.penaltyExcessLoad(self.routeU.load + self.params.cli[self.nodeV.cour].demand) - self.routeU.penalty
                                   + self.penaltyExcessLoad(self.routeV.load - self.params.cli[self.nodeV.cour].demand) - self.routeV.penalty
                                   + self.penaltyExcessDuration(self.routeU.duration + deltaDistRouteU + self.params.cli[self.nodeV.cour].serviceDuration)
                                   + self.penaltyExcessDuration(self.routeV.duration + deltaDistRouteV - self.params.cli[self.nodeV.cour].serviceDuration))
            if mySwapStar.moveCost < myBestSwapStar.moveCost:
                myBestSwapStar = mySwapStar
            self.nodeV = self.nodeV.next

        if myBestSwapStar.moveCost > -MY_EPSILON:
            return False

        if myBestSwapStar.bestPositionU is not None:
            self.insertNode(myBestSwapStar.U, myBestSwapStar.bestPositionU)
        if myBestSwapStar.bestPositionV is not None:
            self.insertNode(myBestSwapStar.V, myBestSwapStar.bestPositionV)
        self.nbMoves += 1
        self.searchCompleted = False
        self.updateRouteData(self.routeU)
        self.updateRouteData(self.routeV)
        return True

    def getCheapestInsertSimultRemoval(self, U: Node, V: Node) -> float:
        myBestInsert = self.bestInsertClient[V.route.cour][U.cour]

        bestPosition = myBestInsert.bestLocation[0]
        bestCost = myBestInsert.bestCost[0]
        found = (bestPosition != V and bestPosition.next != V)

        if not found and myBestInsert.bestLocation[1] is not None:
            bestPosition = myBestInsert.bestLocation[1]
            bestCost = myBestInsert.bestCost[1]
            found = (bestPosition != V and bestPosition.next != V)
            if not found and myBestInsert.bestLocation[2] is not None:
                bestPosition = myBestInsert.bestLocation[2]
                bestCost = myBestInsert.bestCost[2]
                found = True

        deltaCost = (self.params.timeCost[V.prev.cour][U.cour]
                     + self.params.timeCost[U.cour][V.next.cour]
                     - self.params.timeCost[V.prev.cour][V.next.cour])
        if not found or deltaCost < bestCost:
            bestPosition = V.prev
            bestCost = deltaCost

        return bestCost, bestPosition

    def preprocessInsertions(self, R1: Route, R2: Route):
        U = R1.depot.next
        while not U.isDepot:
            U.deltaRemoval = (self.params.timeCost[U.prev.cour][U.next.cour]
                              - self.params.timeCost[U.prev.cour][U.cour]
                              - self.params.timeCost[U.cour][U.next.cour])
            if R2.whenLastModified > self.bestInsertClient[R2.cour][U.cour].whenLastCalculated:
                self.bestInsertClient[R2.cour][U.cour].reset()
                self.bestInsertClient[R2.cour][U.cour].whenLastCalculated = self.nbMoves
                self.bestInsertClient[R2.cour][U.cour].bestCost[0] = (self.params.timeCost[0][U.cour]
                                                                       + self.params.timeCost[U.cour][R2.depot.next.cour]
                                                                       - self.params.timeCost[0][R2.depot.next.cour])
                self.bestInsertClient[R2.cour][U.cour].bestLocation[0] = R2.depot
                V = R2.depot.next
                while not V.isDepot:
                    deltaCost = (self.params.timeCost[V.cour][U.cour]
                                 + self.params.timeCost[U.cour][V.next.cour]
                                 - self.params.timeCost[V.cour][V.next.cour])
                    self.bestInsertClient[R2.cour][U.cour].compareAndAdd(deltaCost, V)
                    V = V.next
            U = U.next

    @staticmethod
    def insertNode(U: Node, V: Node):
        U.prev.next = U.next
        U.next.prev = U.prev
        V.next.prev = U
        U.prev = V
        U.next = V.next
        V.next = U
        U.route = V.route

    @staticmethod
    def swapNode(U: Node, V: Node):
        myVPred = V.prev
        myVSuiv = V.next
        myUPred = U.prev
        myUSuiv = U.next
        myRouteU = U.route
        myRouteV = V.route

        myUPred.next = V
        myUSuiv.prev = V
        myVPred.next = U
        myVSuiv.prev = U

        U.prev = myVPred
        U.next = myVSuiv
        V.prev = myUPred
        V.next = myUSuiv

        U.route = myRouteV
        V.route = myRouteU
    # here
    def updateRouteData(self, myRoute: Route):
        myplace = 0
        myload = 0.0
        mytime = 0.0
        myReversalDistance = 0.0
        cumulatedX = 0.0
        cumulatedY = 0.0

        mynode = myRoute.depot
        mynode.position = 0
        mynode.cumulatedLoad = 0.0
        mynode.cumulatedTime = 0.0
        mynode.cumulatedReversalDistance = 0.0

        firstIt = True
        while not mynode.isDepot or firstIt:
            mynode = mynode.next
            myplace += 1
            mynode.position = myplace
            myload += self.params.cli[mynode.cour].demand
            mytime += self.params.timeCost[mynode.prev.cour][mynode.cour] + self.params.cli[mynode.cour].serviceDuration
            myReversalDistance += self.params.timeCost[mynode.cour][mynode.prev.cour] - self.params.timeCost[mynode.prev.cour][mynode.cour]
            mynode.cumulatedLoad = myload
            mynode.cumulatedTime = mytime
            mynode.cumulatedReversalDistance = myReversalDistance
            if not mynode.isDepot:
                cumulatedX += self.params.cli[mynode.cour].coordX
                cumulatedY += self.params.cli[mynode.cour].coordY
                if firstIt:
                    myRoute.sector.initialize(self.params.cli[mynode.cour].polarAngle)
                else:
                    myRoute.sector.extend(self.params.cli[mynode.cour].polarAngle)
            firstIt = False

        myRoute.duration = mytime
        myRoute.load = myload
        myRoute.penalty = self.penaltyExcessDuration(mytime) + self.penaltyExcessLoad(myload)
        myRoute.nbCustomers = myplace - 1
        myRoute.reversalDistance = myReversalDistance
        myRoute.whenLastModified = self.nbMoves

        if myRoute.nbCustomers == 0:
            myRoute.polarAngleBarycenter = 1e30
            self.emptyRoutes.add(myRoute.cour)
        else:
            myRoute.polarAngleBarycenter = math.atan2(cumulatedY / myRoute.nbCustomers - self.params.cli[0].coordY,
                                                       cumulatedX / myRoute.nbCustomers - self.params.cli[0].coordX)
            self.emptyRoutes.discard(myRoute.cour)

    

    def loadIndividual(self, indiv: Individual):
        self.emptyRoutes.clear()
        self.nbMoves = 0
        for r in range(self.params.nbVehicles):
            myDepot = self.depots[r]
            myDepotFin = self.depotsEnd[r]
            myRoute = self.routes[r]
            myDepot.prev = myDepotFin
            myDepotFin.next = myDepot
            if indiv.chromR[r]:
                myClient = self.clients[indiv.chromR[r][0]]
                myClient.route = myRoute
                myClient.prev = myDepot
                myDepot.next = myClient
                for i in range(1, len(indiv.chromR[r])):
                    myClientPred = myClient
                    myClient = self.clients[indiv.chromR[r][i]]
                    myClient.prev = myClientPred
                    myClientPred.next = myClient
                    myClient.route = myRoute
                myClient.next = myDepotFin
                myDepotFin.prev = myClient
            else:
                myDepot.next = myDepotFin
                myDepotFin.prev = myDepot
            self.updateRouteData(self.routes[r])
            self.routes[r].whenLastTestedSWAPStar = -1
            for i in range(1, self.params.nbClients + 1):
                self.bestInsertClient[r][i].whenLastCalculated = -1

        for i in range(1, self.params.nbClients + 1):
            self.clients[i].whenLastTestedRI = -1

    def exportIndividual(self, indiv: Individual):
        routePolarAngles = [(self.routes[r].polarAngleBarycenter, r) for r in range(self.params.nbVehicles)]
        routePolarAngles.sort()

        pos = 0
        for r in range(self.params.nbVehicles):
            indiv.chromR[r].clear()
            node = self.depots[routePolarAngles[r][1]].next
            while not node.isDepot:
                indiv.chromT[pos] = node.cour
                indiv.chromR[r].append(node.cour)
                node = node.next
                pos += 1

        indiv.evaluateCompleteCost(self.params)


# ================================================================
# Population
# ================================================================
class Population:
    def __init__(self, params: Params, split: Split, localSearch: LocalSearch):
        self.params = params
        self.split = split
        self.localSearch = localSearch
        self.feasibleSubpop = []  # List[Individual]
        self.infeasibleSubpop = []
        self.listFeasibilityLoad = [True] * params.ap.nbIterPenaltyManagement
        self.listFeasibilityDuration = [True] * params.ap.nbIterPenaltyManagement
        self.searchProgress = []  # List[Tuple[float, float]] (time, cost)
        self.bestSolutionRestart = Individual(params)
        self.bestSolutionOverall = Individual(params)

    def generatePopulation(self):
        if self.params.verbose:
            print("----- BUILDING INITIAL POPULATION")
        i = 0
        while i < 4 * self.params.ap.mu and (i == 0 or self.params.ap.timeLimit == 0 or clock() - self.params.startTime < self.params.ap.timeLimit):
            randomIndiv = Individual(self.params)
            self.split.generalSplit(randomIndiv, self.params.nbVehicles)
            self.localSearch.run(randomIndiv, self.params.penaltyCapacity, self.params.penaltyDuration)
            self.addIndividual(randomIndiv, True)
            if not randomIndiv.eval.isFeasible and self.params.ran.randint(0, 1) == 0:
                self.localSearch.run(randomIndiv, self.params.penaltyCapacity * 10.0, self.params.penaltyDuration * 10.0)
                if randomIndiv.eval.isFeasible:
                    self.addIndividual(randomIndiv, False)
            i += 1

    def addIndividual(self, indiv: Individual, updateFeasible: bool) -> bool:
        if updateFeasible:
            self.listFeasibilityLoad.pop(0)
            self.listFeasibilityLoad.append(indiv.eval.capacityExcess < MY_EPSILON)
            self.listFeasibilityDuration.pop(0)
            self.listFeasibilityDuration.append(indiv.eval.durationExcess < MY_EPSILON)

        subpop = self.feasibleSubpop if indiv.eval.isFeasible else self.infeasibleSubpop

        myIndividual = Individual.__new__(Individual)
        # Shallow copy of the data
        myIndividual.eval = indiv.eval
        myIndividual.chromT = indiv.chromT[:]
        myIndividual.chromR = [r[:] for r in indiv.chromR]
        myIndividual.successors = indiv.successors[:]
        myIndividual.predecessors = indiv.predecessors[:]
        myIndividual.indivsPerProximity = {}
        myIndividual.biasedFitness = 0.0

        for indiv2 in subpop:
            myDistance = self.brokenPairsDistance(myIndividual, indiv2)
            indiv2.indivsPerProximity[myDistance] = myIndividual
            myIndividual.indivsPerProximity[myDistance] = indiv2

        # Insert sorted by penalized cost
        place = len(subpop)
        while place > 0 and subpop[place - 1].eval.penalizedCost > indiv.eval.penalizedCost - MY_EPSILON:
            place -= 1
        subpop.insert(place, myIndividual)

        if len(subpop) > self.params.ap.mu + self.params.ap.lambda_:
            while len(subpop) > self.params.ap.mu:
                self.removeWorstBiasedFitness(subpop)

        isNewBest = False
        if indiv.eval.isFeasible and indiv.eval.penalizedCost < self.bestSolutionRestart.eval.penalizedCost - MY_EPSILON:
            self.bestSolutionRestart = myIndividual
            if indiv.eval.penalizedCost < self.bestSolutionOverall.eval.penalizedCost - MY_EPSILON:
                self.bestSolutionOverall = myIndividual
                self.searchProgress.append((clock() - self.params.startTime, self.bestSolutionOverall.eval.penalizedCost))
            isNewBest = True

        return isNewBest

    def updateBiasedFitnesses(self, pop):
        ranking = []
        for i in range(len(pop)):
            ranking.append((-self.averageBrokenPairsDistanceClosest(pop[i], self.params.ap.nbClose), i))
        ranking.sort()

        if len(pop) == 1:
            pop[0].biasedFitness = 0
        else:
            for i in range(len(pop)):
                divRank = float(i) / float(len(pop) - 1)
                fitRank = float(ranking[i][1]) / float(len(pop) - 1)
                if len(pop) <= self.params.ap.nbElite:
                    pop[ranking[i][1]].biasedFitness = fitRank
                else:
                    pop[ranking[i][1]].biasedFitness = fitRank + (1.0 - float(self.params.ap.nbElite) / float(len(pop))) * divRank

    def removeWorstBiasedFitness(self, pop):
        self.updateBiasedFitnesses(pop)
        if len(pop) <= 1:
            raise RuntimeError("Eliminating the best individual: this should not occur in HGS")

        worstIndividual = None
        worstIndividualPosition = -1
        isWorstIndividualClone = False
        worstIndividualBiasedFitness = -1e30

        for i in range(1, len(pop)):
            isClone = (self.averageBrokenPairsDistanceClosest(pop[i], 1) < MY_EPSILON)
            if (isClone and not isWorstIndividualClone) or (isClone == isWorstIndividualClone and pop[i].biasedFitness > worstIndividualBiasedFitness):
                worstIndividualBiasedFitness = pop[i].biasedFitness
                isWorstIndividualClone = isClone
                worstIndividualPosition = i
                worstIndividual = pop[i]

        pop.pop(worstIndividualPosition)

        for indiv2 in pop:
            keys_to_remove = [k for k, v in indiv2.indivsPerProximity.items() if v is worstIndividual]
            for k in keys_to_remove:
                del indiv2.indivsPerProximity[k]

    def restart(self):
        if self.params.verbose:
            print("----- RESET: CREATING A NEW POPULATION -----")
        self.feasibleSubpop.clear()
        self.infeasibleSubpop.clear()
        self.bestSolutionRestart = Individual(self.params)
        self.generatePopulation()

    def managePenalties(self):
        fractionFeasibleLoad = float(self.listFeasibilityLoad.count(True)) / float(len(self.listFeasibilityLoad))
        if fractionFeasibleLoad < self.params.ap.targetFeasible - 0.05 and self.params.penaltyCapacity < 100000.0:
            self.params.penaltyCapacity = min(self.params.penaltyCapacity * self.params.ap.penaltyIncrease, 100000.0)
        elif fractionFeasibleLoad > self.params.ap.targetFeasible + 0.05 and self.params.penaltyCapacity > 0.1:
            self.params.penaltyCapacity = max(self.params.penaltyCapacity * self.params.ap.penaltyDecrease, 0.1)

        fractionFeasibleDuration = float(self.listFeasibilityDuration.count(True)) / float(len(self.listFeasibilityDuration))
        if fractionFeasibleDuration < self.params.ap.targetFeasible - 0.05 and self.params.penaltyDuration < 100000.0:
            self.params.penaltyDuration = min(self.params.penaltyDuration * self.params.ap.penaltyIncrease, 100000.0)
        elif fractionFeasibleDuration > self.params.ap.targetFeasible + 0.05 and self.params.penaltyDuration > 0.1:
            self.params.penaltyDuration = max(self.params.penaltyDuration * self.params.ap.penaltyDecrease, 0.1)

        for i in range(len(self.infeasibleSubpop)):
            self.infeasibleSubpop[i].eval.penalizedCost = (self.infeasibleSubpop[i].eval.distance
                                                           + self.params.penaltyCapacity * self.infeasibleSubpop[i].eval.capacityExcess
                                                           + self.params.penaltyDuration * self.infeasibleSubpop[i].eval.durationExcess)

        for i in range(len(self.infeasibleSubpop)):
            for j in range(len(self.infeasibleSubpop) - i - 1):
                if self.infeasibleSubpop[j].eval.penalizedCost > self.infeasibleSubpop[j + 1].eval.penalizedCost + MY_EPSILON:
                    indiv = self.infeasibleSubpop[j]
                    self.infeasibleSubpop[j] = self.infeasibleSubpop[j + 1]
                    self.infeasibleSubpop[j + 1] = indiv

    def getBinaryTournament(self):
        total = len(self.feasibleSubpop) + len(self.infeasibleSubpop)
        place1 = self.params.ran.randint(0, total - 1)
        place2 = self.params.ran.randint(0, total - 1)
        indiv1 = self.feasibleSubpop[place1] if place1 < len(self.feasibleSubpop) else self.infeasibleSubpop[place1 - len(self.feasibleSubpop)]
        indiv2 = self.feasibleSubpop[place2] if place2 < len(self.feasibleSubpop) else self.infeasibleSubpop[place2 - len(self.feasibleSubpop)]
        self.updateBiasedFitnesses(self.feasibleSubpop)
        self.updateBiasedFitnesses(self.infeasibleSubpop)
        return indiv1 if indiv1.biasedFitness < indiv2.biasedFitness else indiv2

    def getBestFeasible(self):
        if self.feasibleSubpop:
            return self.feasibleSubpop[0]
        return None

    def getBestInfeasible(self):
        if self.infeasibleSubpop:
            return self.infeasibleSubpop[0]
        return None

    def getBestFound(self):
        if self.bestSolutionOverall.eval.penalizedCost < 1e29:
            return self.bestSolutionOverall
        return None

    def printState(self, nbIter: int, nbIterNoImprovement: int):
        if self.params.verbose:
            elapsed = clock() - self.params.startTime
            bestFeas = self.getBestFeasible()
            bestInf = self.getBestInfeasible()
            feasLoad = float(self.listFeasibilityLoad.count(True)) / float(len(self.listFeasibilityLoad))
            feasDur = float(self.listFeasibilityDuration.count(True)) / float(len(self.listFeasibilityDuration))

            if bestFeas is not None:
                print(f"It {nbIter:6d} {nbIterNoImprovement:6d} | T(s) {elapsed:.2f} | Feas {len(self.feasibleSubpop):3d} {bestFeas.eval.penalizedCost:.2f} {self.getAverageCost(self.feasibleSubpop):.2f}", end="")
            else:
                print(f"It {nbIter:6d} {nbIterNoImprovement:6d} | T(s) {elapsed:.2f} | NO-FEASIBLE", end="")

            if bestInf is not None:
                print(f" | Inf {len(self.infeasibleSubpop):3d} {bestInf.eval.penalizedCost:.2f} {self.getAverageCost(self.infeasibleSubpop):.2f}", end="")
            else:
                print(f" | NO-INFEASIBLE", end="")

            print(f" | Div {self.getDiversity(self.feasibleSubpop):.2f} {self.getDiversity(self.infeasibleSubpop):.2f}", end="")
            print(f" | Feas {feasLoad:.2f} {feasDur:.2f}", end="")
            print(f" | Pen {self.params.penaltyCapacity:.2f} {self.params.penaltyDuration:.2f}")

    def brokenPairsDistance(self, indiv1: Individual, indiv2: Individual) -> float:
        differences = 0
        for j in range(1, self.params.nbClients + 1):
            if (indiv1.successors[j] != indiv2.successors[j] and
                indiv1.successors[j] != indiv2.predecessors[j]):
                differences += 1
            if (indiv1.predecessors[j] == 0 and indiv2.predecessors[j] != 0 and
                indiv2.successors[j] != 0):
                differences += 1
        return float(differences) / float(self.params.nbClients)

    def averageBrokenPairsDistanceClosest(self, indiv: Individual, nbClosest: int) -> float:
        result = 0.0
        distances = sorted(indiv.indivsPerProximity.keys())
        maxSize = min(nbClosest, len(distances))
        for i in range(maxSize):
            result += distances[i]
        return result / float(maxSize) if maxSize > 0 else 0.0

    def getDiversity(self, pop) -> float:
        size = min(self.params.ap.mu, len(pop))
        if size <= 0:
            return -1.0
        avg = 0.0
        for i in range(size):
            avg += self.averageBrokenPairsDistanceClosest(pop[i], size)
        return avg / float(size)

    def getAverageCost(self, pop) -> float:
        size = min(self.params.ap.mu, len(pop))
        if size <= 0:
            return -1.0
        avg = 0.0
        for i in range(size):
            avg += pop[i].eval.penalizedCost
        return avg / float(size)

    def exportSearchProgress(self, fileName: str, instanceName: str):
        with open(fileName, 'w') as myfile:
            for state in self.searchProgress:
                myfile.write(f"{instanceName};{self.params.ap.seed};{state[1]};{state[0]}\n")

    def exportCVRPLibFormat(self, indiv: Individual, fileName: str):
        with open(fileName, 'w') as myfile:
            for k in range(len(indiv.chromR)):
                if indiv.chromR[k]:
                    myfile.write(f"Route #{k + 1}:")
                    for cust in indiv.chromR[k]:
                        myfile.write(f" {cust}")
                    myfile.write("\n")
            myfile.write(f"Cost {indiv.eval.penalizedCost}\n")


# ================================================================
# Genetic
# ================================================================
class Genetic:
    def __init__(self, params: Params):
        self.params = params
        self.split = Split(params)
        self.localSearch = LocalSearch(params)
        self.population = Population(params, self.split, self.localSearch)
        self.offspring = Individual(params)

    def crossoverOX(self, result: Individual, parent1: Individual, parent2: Individual):
        freqClient = [False] * (self.params.nbClients + 1)

        start = self.params.ran.randint(0, self.params.nbClients - 1)
        end = self.params.ran.randint(0, self.params.nbClients - 1)
        while end == start:
            end = self.params.ran.randint(0, self.params.nbClients - 1)

        j = start
        while j % self.params.nbClients != (end + 1) % self.params.nbClients:
            idx = j % self.params.nbClients
            result.chromT[idx] = parent1.chromT[idx]
            freqClient[result.chromT[idx]] = True
            j += 1

        for i in range(1, self.params.nbClients + 1):
            temp = parent2.chromT[(end + i) % self.params.nbClients]
            if not freqClient[temp]:
                result.chromT[j % self.params.nbClients] = temp
                j += 1

        self.split.generalSplit(result, parent1.eval.nbRoutes)

    def run(self):
        self.population.generatePopulation()

        nbIter = 0
        nbIterNonProd = 1
        if self.params.verbose:
            print("----- STARTING GENETIC ALGORITHM")

        while nbIterNonProd <= self.params.ap.nbIter and (self.params.ap.timeLimit == 0 or clock() - self.params.startTime < self.params.ap.timeLimit):
            self.crossoverOX(self.offspring, self.population.getBinaryTournament(), self.population.getBinaryTournament())

            self.localSearch.run(self.offspring, self.params.penaltyCapacity, self.params.penaltyDuration)
            isNewBest = self.population.addIndividual(self.offspring, True)

            if not self.offspring.eval.isFeasible and self.params.ran.randint(0, 1) == 0:
                self.localSearch.run(self.offspring, self.params.penaltyCapacity * 10.0, self.params.penaltyDuration * 10.0)
                if self.offspring.eval.isFeasible:
                    isNewBest = self.population.addIndividual(self.offspring, False) or isNewBest

            if isNewBest:
                nbIterNonProd = 1
            else:
                nbIterNonProd += 1

            if nbIter % self.params.ap.nbIterPenaltyManagement == 0:
                self.population.managePenalties()
            if nbIter % self.params.ap.nbIterTraces == 0:
                self.population.printState(nbIter, nbIterNonProd)

            if self.params.ap.timeLimit != 0 and nbIterNonProd == self.params.ap.nbIter:
                self.population.restart()
                nbIterNonProd = 1

            nbIter += 1

        if self.params.verbose:
            elapsed = clock() - self.params.startTime
            print(f"----- GENETIC ALGORITHM FINISHED AFTER {nbIter} ITERATIONS. TIME SPENT: {elapsed}")


# ================================================================
# InstanceCVRPLIB
# ================================================================
class InstanceCVRPLIB:
    def __init__(self, pathToInstance: str, isRoundingInteger: bool = False):
        self.x_coords = []
        self.y_coords = []
        self.dist_mtx = []
        self.service_time = []
        self.demands = []
        self.durationLimit = 1e30
        self.vehicleCapacity = 1e30
        self.isDurationConstraint = False
        self.nbClients = 0

        with open(pathToInstance, 'r') as inputFile:
            lines = inputFile.readlines()

        i = 0
        # Skip initial header lines
        while i < len(lines) and "NODE_COORD_SECTION" not in lines[i]:
            line = lines[i].strip()
            if line.startswith("DIMENSION"):
                parts = line.split()
                self.nbClients = int(parts[-1]) - 1
            elif line.startswith("CAPACITY"):
                parts = line.split()
                self.vehicleCapacity = float(parts[-1])
            elif line.startswith("DISTANCE"):
                parts = line.split()
                self.durationLimit = float(parts[-1])
                self.isDurationConstraint = True
            elif line.startswith("SERVICE_TIME"):
                parts = line.split()
                serviceTimeData = float(parts[-1])
            elif "EDGE_WEIGHT_TYPE" in line:
                pass
            i += 1

        if self.nbClients <= 0:
            raise RuntimeError("Number of nodes is undefined")
        if self.vehicleCapacity == 1e30:
            raise RuntimeError("Vehicle capacity is undefined")

        # Find NODE_COORD_SECTION
        while i < len(lines) and "NODE_COORD_SECTION" not in lines[i]:
            i += 1
        i += 1  # Skip NODE_COORD_SECTION header

        self.x_coords = [0.0] * (self.nbClients + 1)
        self.y_coords = [0.0] * (self.nbClients + 1)
        self.demands = [0.0] * (self.nbClients + 1)
        self.service_time = [0.0] * (self.nbClients + 1)

        # Read coordinates
        for idx in range(self.nbClients + 1):
            parts = lines[i].strip().split()
            node_number = int(parts[0])
            if node_number != idx + 1:
                raise RuntimeError("The node numbering is not in order.")
            self.x_coords[idx] = float(parts[1])
            self.y_coords[idx] = float(parts[2])
            i += 1

        # Find DEMAND_SECTION
        while i < len(lines) and "DEMAND_SECTION" not in lines[i]:
            i += 1
        if i >= len(lines):
            raise RuntimeError("DEMAND_SECTION not found")
        i += 1  # Skip DEMAND_SECTION header

        for idx in range(self.nbClients + 1):
            parts = lines[i].strip().split()
            self.demands[idx] = float(parts[-1])
            self.service_time[idx] = 0.0 if idx == 0 else 0.0  # serviceTimeData set above
            i += 1

        # Calculate distance matrix
        self.dist_mtx = [[0.0] * (self.nbClients + 1) for _ in range(self.nbClients + 1)]
        for ii in range(self.nbClients + 1):
            for jj in range(self.nbClients + 1):
                dx = self.x_coords[ii] - self.x_coords[jj]
                dy = self.y_coords[ii] - self.y_coords[jj]
                self.dist_mtx[ii][jj] = math.sqrt(dx * dx + dy * dy)
                if isRoundingInteger:
                    self.dist_mtx[ii][jj] = round(self.dist_mtx[ii][jj])

        # Verify DEPOT_SECTION (last line should be "DEPOT_SECTION     1       EOF")
        # Not critical, skip validation

# EVOLVE-BLOCK-END

# ================================================================
# Main entry point
# ================================================================
def solve(path: str, rounding: bool=True):
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

    try:
        cvrp = InstanceCVRPLIB(path, isRoundingInteger=rounding)
        ap = AlgorithmParameters()
        ap.timeLimit = 10

        params = Params(cvrp.x_coords, cvrp.y_coords, cvrp.dist_mtx,
                        cvrp.service_time, cvrp.demands,
                        cvrp.vehicleCapacity, cvrp.durationLimit,
                        10**9, cvrp.isDurationConstraint,
                        False, ap)

        solver = Genetic(params)
        solver.run()

        sol = solver.population.getBestFound()
        return [route for route in sol.chromR if route]

    except RuntimeError as e:
        print(f"EXCEPTION | {e}")
    except Exception as e:
        print(f"EXCEPTION | {e}")

