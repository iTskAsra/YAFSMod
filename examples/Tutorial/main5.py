import random
import networkx as nx
import argparse
from pathlib import Path
import time
import numpy as np

from yafs.core import Sim
from yafs.application import Application, Message
from yafs.population import *
from yafs.topology import Topology

from simpleSelection import MinimunPath
from simplePlacement import CloudPlacement
from yafs.stats import Stats
from yafs.distribution import deterministic_distribution, exponential_distribution
from yafs.application import fractional_selectivity

RANDOM_SEED = 1

def create_complex_application():
    # APPLICATION
    a = Application(name="ComplexApp")

    # Modules (S) --> (ServiceA) --> (ServiceB) --> (ServiceC) --> (A)
    a.set_modules([{"Sensor":{"Type":Application.TYPE_SOURCE}},
                   {"ServiceA": {"RAM": 100, "Type": Application.TYPE_MODULE}},
                   {"ServiceB": {"RAM": 200, "Type": Application.TYPE_MODULE}},
                   {"ServiceC": {"RAM": 150, "Type": Application.TYPE_MODULE}},
                   {"Actuator": {"Type": Application.TYPE_SINK}}
                   ])

    # Messages among MODULES
    m_a = Message("M.A", "Sensor", "ServiceA", instructions=100*10**6, bytes=5000)
    m_b = Message("M.B", "ServiceA", "ServiceB", instructions=200*10**6, bytes=10000)
    m_c = Message("M.C", "ServiceB", "ServiceC", instructions=150*10**6, bytes=7000)
    m_d = Message("M.D", "ServiceC", "Actuator", instructions=100*10**6, bytes=5000)

    # Source messages
    a.add_source_messages(m_a)

    # MODULES/SERVICES: Definition of Generators and Consumers
    a.add_service_module("ServiceA", m_a, m_b, fractional_selectivity, threshold=1.0)
    a.add_service_module("ServiceB", m_b, m_c, fractional_selectivity, threshold=1.0)
    a.add_service_module("ServiceC", m_c, m_d, fractional_selectivity, threshold=1.0)

    return a

def create_json_topology():
    # TOPOLOGY DEFINITION
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

def main(simulated_time):

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    folder_results = Path("results/")
    folder_results.mkdir(parents=True, exist_ok=True)
    folder_results = str(folder_results) + "/"

    # TOPOLOGY from a json
    t = Topology()
    t_json = create_json_topology()
    t.load(t_json)
    nx.write_gexf(t.G, folder_results + "graph_main1") 

    # APPLICATION
    app = create_complex_application()

    # PLACEMENT algorithm
    placement = CloudPlacement("onCloud") 
    placement.scaleService({"ServiceA": 1, "ServiceB": 1, "ServiceC": 1})

    # POPULATION algorithm
    pop = Statical("Statical")
    pop.set_sink_control({"model": "actuator-device", "number": 1, "module": app.get_sink_modules()})
    dDistribution = deterministic_distribution(name="Deterministic", time=100)
    pop.set_src_control({"model": "sensor-device", "number": 1, "message": app.get_message("M.A"), "distribution": dDistribution})

    # SELECTOR algorithm
    selectorPath = MinimunPath()

    # SIMULATION ENGINE
    stop_time = simulated_time
    s = Sim(t, default_results_path=folder_results + "sim_trace")
    s.deploy_app2(app, placement, pop, selectorPath)

    # RUNNING - last step
    s.run(stop_time, show_progress_monitor=False)
    s.print_debug_assignaments()

    stats = Stats(defaultPath=folder_results + "sim_trace")
    stats.compute_times_df()
    print(stats.df)

if __name__ == '__main__':
    import logging.config
    import os

    logging.config.fileConfig(os.getcwd() + '/logging.ini')

    start_time = time.time()
    main(simulated_time=10000)

    print("\n--- %s seconds ---" % (time.time() - start_time))