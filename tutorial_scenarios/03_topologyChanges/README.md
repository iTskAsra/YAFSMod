One way to run the code:

```
export PYTHONPATH=$PYTHONPATH:~/YAFS/src/
cd YAFS/tutorial_scenarios/03_topologyChanges/
python main.py
```


The project contains the next files:

```
├── data
│   ├── allocDefinition.json
│   ├── appDefinition.json
│   └── usersDefinition.json
├── logging.ini
├── main.py
├── readme.md
└── results
    ├── graph_binomial_tree_5
    ├── sim_trace.csv
    ├── sim_trace_link.csv
    └── theNew_topology
```

- main.py controls this simulation and generates all the pieces.
- data/allocDefinition.json defines the allocation of app's instances
- data/appDefinition.json defines the description of the applications
- data/usersDefinition.json defines the allocation of the "users" 
- results/graph_binomial_tree_5 a figure of the resulting topology using NetworkX functions.
- results/sim_trace.csv the simulation traces, it contains the requests handled by each instance along the simulation.
- results/sim_trace_link.csv the simlation traces, it contains the network messages between the nodes generated by the requests.
- results/theNew_topology a figure with the new topology, you can use gephi to open it.  
  
### The changes on the topology
They are defined by the next code in main.py file:

```
    """
    """
    This internal monitor in the simulator (a DES process) changes the sim's behaviour. 
    You can have multiples monitors doing different or same tasks.
    
    In this case: it changes the topology.
    """
    dist = deterministicDistributionStartPoint(stop_time/4., stop_time/2.0/10.0, name="Deterministic")
    evol = CustomStrategy(folder_results)
    s.deploy_monitor("CrazyTopology",
                     evol,
                     dist,
                     **{"sim": s, "routing": selectorPath}) # __call__ args 
```