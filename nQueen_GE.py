import numpy as np
import random
import time

# Define parameter sets for the Genetic Algorithm with different levels of intensity and search space.
GA_PARAMETER_SETS = {
    1: {"population_size": 50, "max_generations": 500, "crossover_rate": 0.9, "mutation_rate": 0.2},
    2: {"population_size": 100, "max_generations": 600, "crossover_rate": 0.8, "mutation_rate": 0.25},
    3: {"population_size": 150, "max_generations": 1500, "crossover_rate": 0.7, "mutation_rate": 0.3}
}

# Function to calculate the fitness (number of non-attacking pairs of queens)
def fitness(chromosome):
    n = len(chromosome)
    max_fitness = (n*(n-1)) // 2
    horizontal_collisions = sum([chromosome.count(queen)-1 for queen in chromosome]) / 2
    diagonal_collisions = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(chromosome[i] - chromosome[j]) == j - i:
                diagonal_collisions += 1
    return max_fitness - (horizontal_collisions + diagonal_collisions)

# Function to create a random chromosome or use the user-provided initial state
def create_initial_state(n, user_input=None):
    if user_input:
        return [x-1 for x in user_input]
    else:
        return list(np.random.permutation(n))

# Selection function to choose parents for the next generation
def select_parents(population, fitnesses):
    return random.choices(population, weights=fitnesses, k=2)

# Crossover function to produce offspring from two parents
def crossover(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        crossover_point = random.randint(1, len(parent1) - 2)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    else:
        return parent1, parent2

# Mutation function to introduce small changes in the offspring
def mutate(chromosome, mutation_rate):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(chromosome) - 1)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome

# Function to check if a solution is valid
def is_solution_valid(chromosome):
    n = len(chromosome)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(chromosome[i] - chromosome[j]) == j - i:
                return False
    return True

# Function to perform local search on a chromosome using hill climbing
def local_search(chromosome, iterations=10):
    current_fitness = fitness(chromosome)
    for _ in range(iterations):
        new_chromosome = mutate(chromosome[:], 1)  # Full mutation for local search
        new_fitness = fitness(new_chromosome)
        if new_fitness > current_fitness:
            chromosome, current_fitness = new_chromosome, new_fitness
    return chromosome

# Main function implementing the Genetic Algorithm for solving the N-Queens problem
def genetic_algorithm(n, population_size, max_generations, crossover_rate, mutation_rate, initial_state=None, runs=1):
    best_solution_overall = None
    best_fitness_overall = -1

    for run in range(runs):
        population = [create_initial_state(n, initial_state) for _ in range(population_size)]
        best_solution = None
        best_fitness = -1

        for generation in range(max_generations):
            # Apply local search to each chromosome in the population
            population = [local_search(chromosome) for chromosome in population]
            
            fitnesses = [fitness(chromosome) for chromosome in population]
            # Ensure there are no zero or negative fitness values
            if all(f <= 0 for f in fitnesses):
                print("All chromosomes have zero or negative fitness. Adjust mutation or fitness calculation.")
                return None

            # Sort the population by fitness in descending order
            sorted_population = sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)
            population = [x for _, x in sorted_population]
            fitnesses.sort(reverse=True)

            best_current_fitness = fitnesses[0]
            if best_current_fitness > best_fitness:
                best_fitness = best_current_fitness
                best_solution = population[0]

            if is_solution_valid(best_solution) and best_fitness == (n*(n-1))//2:
                break  # Stop if a valid solution is found

            new_population = []
            while len(new_population) < population_size:
                # Ensure selection is possible
                if sum(fitnesses) == 0:
                    # Adjust the fitnesses to allow selection
                    fitnesses = [f + 1 for f in fitnesses]
                parent1, parent2 = select_parents(population, fitnesses)
                child1, child2 = crossover(parent1, parent2, crossover_rate)
                new_population.append(mutate(child1, mutation_rate))
                if len(new_population) < population_size:
                    new_population.append(mutate(child2, mutation_rate))

            population = new_population[:population_size]  # Ensure the population size remains constant

        if best_solution and is_solution_valid(best_solution):
            if best_fitness > best_fitness_overall:
                best_solution_overall = best_solution
                best_fitness_overall = best_fitness
            
        else:
            pass

    return best_solution_overall


# Function to get user input for initial positions or generate random positions
def get_user_input(n):
    choice = input("Do you want to input initial positions? Write 'yes' to input or 'no' to use random initial positions (yes/no): ").strip().lower()
    if choice == 'yes':
        while True:
            try:
                initial_positions = input(f"Enter the initial positions of the queens for a {n}x{n} board, separated by spaces (1-{n}): ")
                positions = list(map(int, initial_positions.split()))
                if len(positions) != n or not all(1 <= pos <= n for pos in positions):
                    raise ValueError("Please enter valid positions for the queens.")
                return positions  # Return the positions as a list of integers
            except ValueError as e:
                print(f"Invalid input. {e} Please enter {n} numbers from 1 to {n}, representing the row positions of the queens.")
    elif choice == 'no':
        return None  # None indicates that a random initial state should be generated
    else:
        print("Invalid choice. Please enter 'yes' or 'no'.")
        return get_user_input(n)

# Example usage of the algorithm
if __name__ == "__main__":
    n = int(input("Enter the board size (n): "))
    while n < 4:
        print("Board size must be 4 or greater.")
        n = int(input("Enter the board size (n): "))

    initial_positions = get_user_input(n)

    params = None
    while True:
        print("Choose a parameter set for the Genetic Algorithm:")
        for key, value in GA_PARAMETER_SETS.items():
            print(f"{key}: {value}")
        print("4: Custom input")
        choice = input("Enter your choice (1/2/3/4): ")
        if choice in ('1', '2', '3'):
            params = GA_PARAMETER_SETS[int(choice)]
            break
        elif choice == '4':
            population_size = int(input("Enter population size: "))
            max_generations = int(input("Enter max generations: "))
            crossover_rate = float(input("Enter crossover rate (0-1): "))
            mutation_rate = float(input("Enter mutation rate (0-1): "))
            params = {
                "population_size": population_size,
                "max_generations": max_generations,
                "crossover_rate": crossover_rate,
                "mutation_rate": mutation_rate
            }
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

    runs = int(input("How many times would you like to run the algorithm? "))
    start_time = time.time()
    best_solution = genetic_algorithm(n, **params, initial_state=initial_positions, runs=runs)
    elapsed_time = time.time() - start_time

    if best_solution:
        best_solution_printable = [x + 1 for x in best_solution]
        print(f"Best solution found: {best_solution_printable}")
    else:
        print("Failed to find a solution.")
    print(f"Elapsed time: {elapsed_time:.2f} seconds.")
