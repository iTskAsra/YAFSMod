import random
import networkx as nx
import argparse
from pathlib import Path
import time
import numpy as np
from deap import base, creator, tools, algorithms

from yafs.core import Sim
from yafs.application import Application, Message
from yafs.population import *
from yafs.topology import Topology
from yafs.placement import Placement
from yafs.selection import Selection

from simpleSelection import MinimunPath
from simplePlacement import CloudPlacement
from yafs.stats import Stats
from yafs.distribution import deterministic_distribution
from yafs.application import fractional_selectivity

RANDOM_SEED = 1

def create_complex_application():
    a = Application(name="ComplexApp")
    a.set_modules([{"Sensor": {"Type": Application.TYPE_SOURCE}},
                   {"ServiceA": {"RAM": 100, "Type": Application.TYPE_MODULE}},
                   {"ServiceB": {"RAM": 200, "Type": Application.TYPE_MODULE}},
                   {"ServiceC": {"RAM": 150, "Type": Application.TYPE_MODULE}},
                   {"Actuator": {"Type": Application.TYPE_SINK}}
                   ])
    m_a = Message("M.A", "Sensor", "ServiceA", instructions=100*10**6, bytes=5000)
    m_b = Message("M.B", "ServiceA", "ServiceB", instructions=200*10**6, bytes=10000)
    m_c = Message("M.C", "ServiceB", "ServiceC", instructions=150*10**6, bytes=7000)
    m_d = Message("M.D", "ServiceC", "Actuator", instructions=100*10**6, bytes=5000)
    a.add_source_messages(m_a)
    a.add_service_module("ServiceA", m_a, m_b, fractional_selectivity, threshold=1.0)
    a.add_service_module("ServiceB", m_b, m_c, fractional_selectivity, threshold=1.0)
    a.add_service_module("ServiceC", m_c, m_d, fractional_selectivity, threshold=1.0)
    return a

def create_json_topology():
    topology_json = {}
    topology_json["entity"] = []
    topology_json["link"] = []

    cloud_dev    = {"id": 0, "model": "cloud","mytag":"cloud", "IPT": 5000 * 10 ** 6, "RAM": 40000,"COST": 3,"WATT":20.0}
    sensor_dev   = {"id": 1, "model": "sensor-device", "IPT": 100* 10 ** 6, "RAM": 4000,"COST": 3,"WATT":40.0}
    actuator_dev = {"id": 2, "model": "actuator-device", "IPT": 100 * 10 ** 6, "RAM": 4000,"COST": 3, "WATT": 40.0}
    fog_dev1 = {"id": 3, "model": "fog-device", "IPT": 2000 * 10 ** 6, "RAM": 16000, "COST": 2, "WATT": 15.0}
    fog_dev2 = {"id": 4, "model": "fog-device", "IPT": 1500 * 10 ** 6, "RAM": 8000, "COST": 2, "WATT": 15.0}

    link1 = {"s": 0, "d": 1, "BW": 10, "PR": 10}
    link2 = {"s": 0, "d": 2, "BW": 10, "PR": 1}
    link3 = {"s": 0, "d": 3, "BW": 20, "PR": 5}
    link4 = {"s": 3, "d": 4, "BW": 20, "PR": 5}
    link5 = {"s": 4, "d": 1, "BW": 10, "PR": 10}
    link6 = {"s": 4, "d": 2, "BW": 10, "PR": 10}

    topology_json["entity"].append(cloud_dev)
    topology_json["entity"].append(sensor_dev)
    topology_json["entity"].append(actuator_dev)
    topology_json["entity"].append(fog_dev1)
    topology_json["entity"].append(fog_dev2)

    topology_json["link"].append(link1)
    topology_json["link"].append(link2)
    topology_json["link"].append(link3)
    topology_json["link"].append(link4)
    topology_json["link"].append(link5)
    topology_json["link"].append(link6)

    return topology_json

class NSGAIIPlacement(Placement):
    def __init__(self, scaleServices, ind):
        self.ind = ind
        super(NSGAIIPlacement, self).__init__("NSGAII")
        self.scaleServices = scaleServices

    def initial_allocation(self, sim, app_name):
        id_cluster = sim.topology.find_IDs({"mytag": "cloud"})[0]
        app = sim.apps[app_name]
        services = app.services
        for i, module in enumerate(services):
            if module in self.scaleServices:
                idDES = int(self.ind[i] * len(sim.topology.G.nodes))  # Scale individual value to node ID
                sim.deploy_module(app_name, module, services[module], [idDES])

class NSGAIISelector(Selection):
    def __init__(self, ind):
        self.ind = ind

    def get_path(self, sim, app_name, message, topology_src, alloc_DES, alloc_module, traffic):
        node_src = topology_src
        DES_dst = alloc_module[app_name][message.dst]
        path = list(nx.shortest_path(sim.topology.G, source=node_src, target=alloc_DES[int(self.ind[0] * len(alloc_DES))]))
        return [path], [alloc_DES[int(self.ind[0] * len(alloc_DES))]]

def evaluate(individual):
    simulated_time = 1000
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    folder_results = Path("results/")
    folder_results.mkdir(parents=True, exist_ok=True)
    folder_results = str(folder_results) + "/"

    t = Topology()
    t_json = create_json_topology()
    t.load(t_json)

    app = create_complex_application()

    placement = NSGAIIPlacement({"ServiceA": 1, "ServiceB": 1, "ServiceC": 1}, individual)
    
    pop = Statical("Statical")
    pop.set_sink_control({"model": "actuator-device", "number": 1, "module": app.get_sink_modules()})
    dDistribution = deterministic_distribution(name="Deterministic", time=100)
    pop.set_src_control({"model": "sensor-device", "number": 1, "message": app.get_message("M.A"), "distribution": dDistribution})

    selectorPath = NSGAIISelector(individual)

    s = Sim(t, default_results_path=folder_results + "sim_trace")
    s.deploy_app2(app, placement, pop, selectorPath)
    s.run(simulated_time, show_progress_monitor=False)

    stats = Stats(defaultPath=folder_results + "sim_trace")
    stats.compute_times_df()

    latency = stats.df["time_latency"].mean()
    energy_consumption = stats.df["time_service"].sum()

    return latency, energy_consumption

def main(simulated_time):
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=5)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", evaluate)

    pop = toolbox.population(n=50)
    ngen = 40
    cxpb, mutpb = 0.7, 0.2

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=50, lambda_=100, cxpb=cxpb, mutpb=mutpb,
                                         ngen=ngen, stats=stats, halloffame=None, verbose=True)

    best_individual = tools.selBest(pop, 1)[0]
    print("Best Individual: ", best_individual)
    print("Best Fitness: ", best_individual.fitness.values)

if __name__ == '__main__':
    import logging.config
    import os

    logging.config.fileConfig(os.getcwd() + '/logging.ini')

    start_time = time.time()
    main(simulated_time=10000)

    print("\n--- %s seconds ---" % (time.time() - start_time))
