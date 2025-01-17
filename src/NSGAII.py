import random
from deap import base, creator, tools, algorithms

# Define your messages, nodes, and modules
# Generate 100 messages, nodes, and modules
messages = ['msg{}'.format(i) for i in range(1, 101)]
nodes = ['node{}'.format(i) for i in range(1, 101)]
modules = ['module{}'.format(i) for i in range(1, 101)]

# Print the first few elements of each list to verify
print("Messages:", messages[:5])
print("Nodes:", nodes[:5])
print("Modules:", modules[:5])


# Create Individual
def create_individual():
    return [(random.choice(messages), random.choice(nodes), random.choice(modules)) for _ in range(len(messages))]

# Fitness and Individual Classes
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))  # Assuming minimization
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Initialize Population
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Fitness Evaluation Function
def evaluate(individual):
    # Implement your evaluation logic here
    # This might involve simulating the network behavior in YAFS based on the individual's decisions
    deadline_penalty = random.random()
    latency_cost = random.random()
    return deadline_penalty, latency_cost

toolbox.register("evaluate", evaluate)

# Genetic Operators
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selNSGA2)

# Genetic Algorithm Parameters
population = toolbox.population(n=50)
ngen = 100  # Number of generations

def custom_mutation(individual):
    # Choose a random element in the individual to mutate
    idx = random.randrange(len(individual))
    # Mutate the chosen element
    # For example, if individual[idx] is a tuple (message, node, module),
    # you might choose to change the node or module
    message, node, module = individual[idx]
    new_node = random.choice(nodes)  # Choose a new node
    new_module = random.choice(modules)  # Choose a new module
    individual[idx] = (message, new_node, new_module)
    return individual,

toolbox.register("mutate", custom_mutation)
for gen in range(ngen):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

# Evaluate Results
best_individuals = tools.selBest(population, k=3)
print("Best Individuals:", best_individuals, best_individuals.__sizeof__())
