import networkx as nx
from yafs.core import Sim
from yafs.topology import Topology
from yafs.application import Application, Message
from yafs.placement import Placement
from yafs.population import Statical
from yafs.selection import Selection

# Define the network topology
def create_topology():
    topology_json = {}
    topology_json["entity"] = []
    topology_json["link"] = []

    cloud_dev = {"id": 0, "model": "cloud", "mytag": "cloud", "IPT": 5000 * 10 ** 6, "RAM": 40000, "COST": 3, "WATT": 20.0}
    sensor_dev = {"id": 1, "model": "sensor-device", "IPT": 100 * 10 ** 6, "RAM": 4000, "COST": 3, "WATT": 40.0}
    actuator_dev = {"id": 2, "model": "actuator-device", "IPT": 100 * 10 ** 6, "RAM": 4000, "COST": 3, "WATT": 40.0}

    link1 = {"s": 0, "d": 1, "BW": 1, "PR": 10}
    link2 = {"s": 0, "d": 2, "BW": 1, "PR": 1}

    topology_json["entity"].append(cloud_dev)
    topology_json["entity"].append(sensor_dev)
    topology_json["entity"].append(actuator_dev)
    topology_json["link"].append(link1)
    topology_json["link"].append(link2)

    t = Topology()
    t.load(topology_json)
    return t

# Define the application
def create_application():
    a = Application(name="SimpleCase")

    a.set_modules([
        {"Sensor": {"Type": Application.TYPE_SOURCE}},
        {"ServiceA": {"RAM": 10, "Type": Application.TYPE_MODULE}},
        {"Actuator": {"Type": Application.TYPE_SINK}}
    ])

    m_a = Message("M.A", "Sensor", "ServiceA", instructions=20 * 10 ** 6, bytes=1000)
    m_b = Message("M.B", "ServiceA", "Actuator", instructions=30 * 10 ** 6, bytes=500)

    a.add_source_messages(m_a)
    a.add_service_module("ServiceA", m_a, m_b, fractional_selectivity=1.0, threshold=1.0)

    return a

# Define the population
def create_population(app):
    pop = Statical("Statical")
    pop.set_src_control(
        {"model": "sensor-device", "number": 1, "message": app.get_message("M.A"), "distribution": "deterministic", "param": {"time_shift": 100}})
    pop.set_sink_control({"model": "actuator-device", "number": 1, "module": app.get_sink_modules()})
    return pop

# Define the placement algorithm
class CloudPlacement(Placement):
    def initial_allocation(self, sim, app_name):
        id_cluster = sim.topology.find_IDs({"mytag": "cloud"})[0]
        app = sim.apps[app_name]
        services = app.services
        for module in services:
            if module in self.scaleServices:
                for rep in range(0, self.scaleServices[module]):
                    sim.deploy_module(app_name, module, services[module], id_cluster)

# Define the path selector
class MinimumPath(Selection):
    def get_path(self, sim, app_name, message, topology_src, alloc_DES, alloc_module, traffic):
        node_src = topology_src
        DES_dst = alloc_module[app_name][message.dst]

        bestPath = []
        bestDES = []

        for des in DES_dst:
            dst_node = alloc_DES[des]
            path = list(nx.shortest_path(sim.topology.G, source=node_src, target=dst_node))
            bestPath = [path]
            bestDES = [des]

        return bestPath, bestDES

# Run the simulation
def main():
    topology = create_topology()
    app = create_application()
    pop = create_population(app)
    
    placement = CloudPlacement("onCloud")
    placement.scaleService({"ServiceA": 1})
    
    selectorPath = MinimumPath()

    sim = Sim(topology)
    sim.deploy_app(app, placement, selectorPath)
    sim.run(1000)  # Run the simulation for a specified amount of time

if __name__ == "__main__":
    main()
