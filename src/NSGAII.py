import random
from deap import base, creator, tools, algorithms

# Create Individual
def create_individual():
    return [random.random() for _ in range(10)]  # Example: 10 random numbers

# Fitness and Individual Classes
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Initialize Population
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Fitness Evaluation Function
def evaluate(individual):
    return sum(individual), 0  # Simple example

toolbox.register("evaluate", evaluate)

# Genetic Operators
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

# Genetic Algorithm Parameters
population = toolbox.population(n=50)
ngen = 10  # Number of generations

# Run the Algorithm
for gen in range(ngen):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

# Evaluate Results
best_individuals = tools.selBest(population, k=3)
print("Best Individuals:", best_individuals)
