import random
import numpy as np
from deap import base, creator, tools, algorithms
from yafs.core import Sim
from yafs.topology import Topology
from yafs.application import Application, Message
from yafs.population import Statical
from yafs.placement import Placement
from yafs.selection import Selection
from yafs.stats import Stats
from yafs.distribution import deterministic_distribution
from yafs.application import fractional_selectivity
from pathlib import Path
import time
import networkx as nx

RANDOM_SEED = 1
POP_SIZE = 50  # Set population size here

# Define the application
def create_complex_application():
    a = Application(name="ComplexApp")
    a.set_modules([{"Sensor": {"Type": Application.TYPE_SOURCE}},
                   {"ServiceA": {"RAM": 100, "Type": Application.TYPE_MODULE}},
                   {"ServiceB": {"RAM": 200, "Type": Application.TYPE_MODULE}},
                   {"ServiceC": {"RAM": 150, "Type": Application.TYPE_MODULE}},
                   {"Actuator": {"Type": Application.TYPE_SINK}}])
    m_a = Message("M.A", "Sensor", "ServiceA", instructions=100*10**6, bytes=5000)
    m_b = Message("M.B", "ServiceA", "ServiceB", instructions=200*10**6, bytes=10000)
    m_c = Message("M.C", "ServiceB", "ServiceC", instructions=150*10**6, bytes=7000)
    m_d = Message("M.D", "ServiceC", "Actuator", instructions=100*10**6, bytes=5000)
    a.add_source_messages(m_a)
    a.add_service_module("ServiceA", m_a, m_b, fractional_selectivity, threshold=1.0)
    a.add_service_module("ServiceB", m_b, m_c, fractional_selectivity, threshold=1.0)
    a.add_service_module("ServiceC", m_c, m_d, fractional_selectivity, threshold=1.0)
    return a

# Define the topology
def create_json_topology():
    """
    Creates a sample topology with nodes and links.
    """
    topology_json = {}
    topology_json["entity"] = []
    topology_json["link"] = []

    # Define nodes with attributes (example: cloud, sensor, actuator, and fog devices)
    cloud_dev = {"id": 0, "model": "cloud", "mytag": "cloud", "IPT": 5000 * 10 ** 6, "RAM": 40000, "COST": 3, "WATT": 20.0}
    sensor_dev = {"id": 1, "model": "sensor-device", "IPT": 100 * 10 ** 6, "RAM": 4000, "COST": 3, "WATT": 40.0}
    actuator_dev = {"id": 2, "model": "actuator-device", "IPT": 100 * 10 ** 6, "RAM": 4000, "COST": 3, "WATT": 40.0}
    fog_dev1 = {"id": 3, "model": "fog-device", "IPT": 2000 * 10 ** 6, "RAM": 16000, "COST": 2, "WATT": 15.0}
    fog_dev2 = {"id": 4, "model": "fog-device", "IPT": 1500 * 10 ** 6, "RAM": 8000, "COST": 2, "WATT": 15.0}
    fog_dev3 = {"id": 5, "model": "fog-device", "IPT": 1800 * 10 ** 6, "RAM": 12000, "COST": 2, "WATT": 15.0}
    fog_dev4 = {"id": 6, "model": "fog-device", "IPT": 1600 * 10 ** 6, "RAM": 10000, "COST": 2, "WATT": 15.0}
    fog_dev5 = {"id": 7, "model": "fog-device", "IPT": 1400 * 10 ** 6, "RAM": 9000, "COST": 2, "WATT": 15.0}
    fog_dev6 = {"id": 8, "model": "fog-device", "IPT": 1300 * 10 ** 6, "RAM": 8500, "COST": 2, "WATT": 15.0}
    fog_dev7 = {"id": 9, "model": "fog-device", "IPT": 1200 * 10 ** 6, "RAM": 8000, "COST": 2, "WATT": 15.0}
    fog_dev8 = {"id": 10, "model": "fog-device", "IPT": 1100 * 10 ** 6, "RAM": 7500, "COST": 2, "WATT": 15.0}
    fog_dev9 = {"id": 11, "model": "fog-device", "IPT": 1000 * 10 ** 6, "RAM": 7000, "COST": 2, "WATT": 15.0}

    # Add nodes to the topology
    topology_json["entity"].append(cloud_dev)
    topology_json["entity"].append(sensor_dev)
    topology_json["entity"].append(actuator_dev)
    topology_json["entity"].append(fog_dev1)
    topology_json["entity"].append(fog_dev2)
    topology_json["entity"].append(fog_dev3)
    topology_json["entity"].append(fog_dev4)
    topology_json["entity"].append(fog_dev5)
    topology_json["entity"].append(fog_dev6)
    topology_json["entity"].append(fog_dev7)
    topology_json["entity"].append(fog_dev8)
    topology_json["entity"].append(fog_dev9)

    # Define links with attributes (source, destination, bandwidth, and propagation delay)
    link1 = {"s": 0, "d": 1, "BW": 10, "PR": 10}
    link2 = {"s": 0, "d": 2, "BW": 10, "PR": 1}
    link3 = {"s": 0, "d": 3, "BW": 20, "PR": 5}
    link4 = {"s": 3, "d": 4, "BW": 20, "PR": 5}
    link5 = {"s": 4, "d": 1, "BW": 10, "PR": 10}
    link6 = {"s": 4, "d": 2, "BW": 10, "PR": 10}
    link7 = {"s": 3, "d": 5, "BW": 15, "PR": 5}
    link8 = {"s": 5, "d": 6, "BW": 15, "PR": 5}
    link9 = {"s": 6, "d": 7, "BW": 15, "PR": 5}
    link10 = {"s": 7, "d": 8, "BW": 15, "PR": 5}
    link11 = {"s": 8, "d": 9, "BW": 15, "PR": 5}
    link12 = {"s": 9, "d": 10, "BW": 15, "PR": 5}
    link13 = {"s": 10, "d": 11, "BW": 15, "PR": 5}

    # Add links to the topology
    topology_json["link"].append(link1)
    topology_json["link"].append(link2)
    topology_json["link"].append(link3)
    topology_json["link"].append(link4)
    topology_json["link"].append(link5)
    topology_json["link"].append(link6)
    topology_json["link"].append(link7)
    topology_json["link"].append(link8)
    topology_json["link"].append(link9)
    topology_json["link"].append(link10)
    topology_json["link"].append(link11)
    topology_json["link"].append(link12)
    topology_json["link"].append(link13)

    return topology_json

# Define DEAP structures
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Function to calculate individual size
def calculate_individual_size(app):
    num_modules = len(app.services)
    num_messages = sum([1 for service in app.services.values() if "to" in service])
    return num_modules + num_messages

# Create initial population
def create_initial_population(pop_size, ind_size):
    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 0, 11)  # Adjust the range as needed for your topology
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=ind_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    return toolbox.population(n=pop_size)

# Generate the initial population
app = create_complex_application()
individual_size = calculate_individual_size(app)
initial_population = create_initial_population(POP_SIZE, individual_size)

# Print some individuals from the initial population
for i, individual in enumerate(initial_population[:5]):
    print(f"Individual {i}: {individual}")

# Define the placement class
class LiteralPlacement(Placement):
    def __init__(self, ind):
        super().__init__("Whatever")
        self.ind = ind

    def initial_allocation(self, sim, app_name):
        app = sim.apps[app_name]
        services = app.services
        num_nodes = len(sim.topology.G.nodes)
        
        for i, module in enumerate(services.keys()):
            if i < len(self.ind):
                print(f"Placing module {module} on node {self.ind[i] % num_nodes}")
                chosen_node = int(self.ind[i]) % num_nodes  # Ensure the chosen node index is valid
                sim.deploy_module(app_name, module, services[module], [chosen_node])

# Define the selection class
class LiteralSelection(Selection):
    def __init__(self, ind, num_paths):
        super().__init__()
        self.ind = ind
        self.num_paths = num_paths

    def get_all_shortest_paths(self, graph, source, target):
        try:
            paths = list(nx.all_shortest_paths(graph, source=source, target=target))
            return paths
        except nx.NetworkXNoPath:
            return []

    def get_path(self, sim, app_name, message, topology_src, alloc_DES, alloc_module, traffic, from_des):
        node_src = topology_src
        DES_dst = alloc_module[app_name][message.dst]
        best_path = []
        best_DES = []
        
        for des in DES_dst:
            dst_node = alloc_DES[des]
            paths = self.get_all_shortest_paths(sim.topology.G, node_src, dst_node)
            num_paths = len(paths)
            if num_paths == 0:
                print(f"No paths found from {node_src} to {dst_node}")
                continue
            if len(self.ind) == 0:
                print(f"No path index available for message {message.name} from {node_src} to {dst_node}")
                chosen_path_index = 0  # Default to the first path if no valid index
            else:
                chosen_path_index = int(self.ind.pop(0)) % num_paths  # Ensure the chosen path index is valid
            
            best_path = [paths[chosen_path_index]]
            best_DES = [des]
            print(f"Path for message {message.name} from {node_src} to {dst_node}: {best_path}")
        
        return best_path, best_DES

# Run YAFS simulation
def run_yafs_simulation(individual):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    folder_results = Path("results/")
    folder_results.mkdir(parents=True, exist_ok=True)
    folder_results = str(folder_results) + "/"

    t = Topology()
    t_json = create_json_topology()
    t.load(t_json)
    num_nodes = len(t_json["entity"])

    app = create_complex_application()

    placement = LiteralPlacement(individual[:len(app.services)])
    pop = Statical("Statical")
    pop.set_sink_control({"model": "actuator-device", "number": 1, "module": app.get_sink_modules()})
    dDistribution = deterministic_distribution(name="Deterministic", time=100)
    pop.set_src_control({"model": "sensor-device", "number": 1, "message": app.get_message("M.A"), "distribution": dDistribution})

    selectorPath = LiteralSelection(individual[len(app.services):], 3)

    s = Sim(t, default_results_path=folder_results + "sim_trace")
    s.deploy_app2(app, placement, pop, selectorPath)
    s.run(1000, show_progress_monitor=False)

    stats = Stats(defaultPath=folder_results + "sim_trace")
    stats.compute_times_df()

    latency = stats.df["time_latency"].mean()
    energy_consumption = stats.df["time_service"].sum()

    return latency, energy_consumption

class NSGAIIOptimizer:
    def __init__(self, pop_size, ngen, cxpb, mutpb):
        self.pop_size = pop_size
        self.ngen = ngen
        self.cxpb = cxpb
        self.mutpb = mutpb

        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_int", random.randint, 0, 11)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_int, n=individual_size)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("evaluate", self.evaluate)

    def evaluate(self, individual):
        latency, energy_consumption = run_yafs_simulation(individual)
        return latency, energy_consumption

    def optimize(self):
        pop = self.toolbox.population(n=self.pop_size)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        pop, log = algorithms.eaMuPlusLambda(pop, self.toolbox, mu=self.pop_size, lambda_=self.pop_size * 2,
                                             cxpb=self.cxpb, mutpb=self.mutpb, ngen=self.ngen, stats=stats, verbose=True)

        best_individual = tools.selBest(pop, 1)[0]
        return best_individual

def main():
    optimizer = NSGAIIOptimizer(pop_size=50, ngen=40, cxpb=0.7, mutpb=0.2)
    best_individual = optimizer.optimize()
    print("Best Individual: ", best_individual)
    print("Best Fitness: ", best_individual.fitness.values)

if __name__ == '__main__':
    import logging.config
    import os

    logging.config.fileConfig(os.getcwd() + '/logging.ini')

    start_time = time.time()
    main()

    print("\n--- %s seconds ---" % (time.time() - start_time))
