import numpy as np

# Define the Rastrigin function
def rastrigin(X):
    A = 10
    return A * len(X) + sum([(x**2 - A * np.cos(2 * np.pi * x)) for x in X])

# Initialize population
def initialize_population(pop_size, dimensions, bounds):
    population = []
    for _ in range(pop_size):
        individual = np.random.uniform(bounds[0], bounds[1], dimensions)
        population.append(individual)
    return population

# Evaluate fitness
def evaluate_population(population):
    return [rastrigin(individual) for individual in population]

# Select parents (roulette wheel selection)
def select_parents(population, fitness, num_parents):
    fitness = np.array(fitness)
    # Invert fitness values for roulette wheel selection
    inverted_fitness = 1 / (fitness + 1e-6)
    probs = inverted_fitness / inverted_fitness.sum()
    selected_indices = np.random.choice(len(population), size=num_parents, p=probs)
    return [population[i] for i in selected_indices]

# Crossover (single-point crossover)
def crossover(parents):
    offspring = []
    for i in range(0, len(parents), 2):
        if i + 1 < len(parents):
            crossover_point = np.random.randint(1, len(parents[i]))
            child1 = np.concatenate((parents[i][:crossover_point], parents[i+1][crossover_point:]))
            child2 = np.concatenate((parents[i+1][:crossover_point], parents[i][crossover_point:]))
            offspring.append(child1)
            offspring.append(child2)
    return offspring

# Mutation (random mutation)
def mutate(offspring, mutation_rate, bounds):
    for individual in offspring:
        if np.random.rand() < mutation_rate:
            mutation_point = np.random.randint(len(individual))
            individual[mutation_point] = np.random.uniform(bounds[0], bounds[1])
    return offspring

# Genetic Algorithm
def genetic_algorithm(pop_size, dimensions, bounds, generations, mutation_rate):
    population = initialize_population(pop_size, dimensions, bounds)
    for generation in range(generations):
        fitness = evaluate_population(population)
        parents = select_parents(population, fitness, pop_size // 2)
        offspring = crossover(parents)
        offspring = mutate(offspring, mutation_rate, bounds)
        population = parents + offspring
        print(f"Generation {generation}: Best fitness = {min(fitness)}")
    best_individual = population[np.argmin(evaluate_population(population))]
    return best_individual

# Parameters
pop_size = 50
dimensions = 5
bounds = [-5.12, 5.12]
generations = 100
mutation_rate = 0.1

# Run GA
best_solution = genetic_algorithm(pop_size, dimensions, bounds, generations, mutation_rate)
print(f"\nBest solution: {best_solution}","\n","Best fitness: {rastrigin(best_solution)}")
