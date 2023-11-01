import numpy as np
from multiprocessing import Pool, cpu_count
import time
import random

# Define sets of parameters for the PSO algorithm
PARAMETER_SETS = {
    1: {"num_particles": 50, "num_iterations": 500, "w": 0.9, "c1": 1.2, "c2": 2.2},
    2: {"num_particles": 100, "num_iterations": 600, "w": 0.8, "c1": 1.3, "c2": 2.1},
    3: {"num_particles": 150, "num_iterations": 700, "w": 0.7, "c1": 1.4, "c2": 2.0}
}

# Particle class represents a solution in the search space
class Particle:
    def __init__(self, initial_position):
        self.position = initial_position.copy()
        np.random.shuffle(self.position)
        self.velocity = np.zeros(len(self.position), dtype=int)
        self.best_position = np.copy(self.position)
        self.best_score = -float('inf')

# Objective function to evaluate the quality of a solution
def objective_function(position):
    n = len(position)
    # Count column conflicts
    column_conflicts = sum(np.bincount(position).clip(0, 1)) - n
    diagonal_conflicts = 0
    # Count diagonal conflicts
    for i in range(n):
        for j in range(i+1, n):
            if position[i] - position[j] == i - j or position[j] - position[i] == i - j:
                diagonal_conflicts += 1
    return -(column_conflicts + diagonal_conflicts)

# Local search function to improve a given solution
def local_search(position):
    n = len(position)
    best_position = position.copy()
    best_score = objective_function(best_position)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(n):
                if i != j:
                    new_position = best_position.copy()
                    new_position[i], new_position[j] = new_position[j], new_position[i]
                    new_score = objective_function(new_position)
                    if new_score > best_score:
                        best_position = new_position.copy()
                        best_score = new_score
                        improved = True
    return best_position

# PSO algorithm implementation
def PSO(num_particles, dimension, num_iterations, w, c1, c2, num_runs, initial_position):
    best_solutions = []
    # Create a multiprocessing pool to parallelize objective function evaluations
    pool = Pool(processes=cpu_count())

    # Diversification: Variable Inertia Weight
    INITIAL_W = w
    FINAL_W = 0.4

    for run in range(num_runs):
        # Initialize particles
        particles = [Particle(initial_position) for _ in range(num_particles)]
        # Determine the global best position
        g_best_position = max(particles, key=lambda p: objective_function(p.position)).position.copy()
        g_best_score = objective_function(g_best_position)
        
        # Initialize previous_g_best_score
        previous_g_best_score = -float('inf')

        # Diversification: Counter for iterations without improvement
        no_improvement_counter = 0
        MAX_ITER_WITHOUT_IMPROVEMENT = 50

        for iteration in range(num_iterations):
            # Update inertia weight dynamically
            current_iteration_fraction = iteration / num_iterations
            w = INITIAL_W - current_iteration_fraction * (INITIAL_W - FINAL_W)

            # Evaluate the objective function for all particles
            scores = pool.map(objective_function, [p.position for p in particles])
            for particle, score in zip(particles, scores):
                # Update particle velocity and position
                inertia = w * particle.velocity
                personal_attraction = c1 * np.random.random() * (particle.best_position - particle.position)
                social_attraction = c2 * np.random.random() * (g_best_position - particle.position)
                particle.velocity = inertia + personal_attraction + social_attraction
                swap_idx1 = int(abs(particle.velocity[0]) % dimension)
                swap_idx2 = int(abs(particle.velocity[1]) % dimension)
                particle.position[swap_idx1], particle.position[swap_idx2] = particle.position[swap_idx2], particle.position[swap_idx1]
                particle.position = particle.position % dimension

                # Update best positions and scores
                particle_score = objective_function(particle.position)
                if particle_score > particle.best_score:
                    particle.best_score = particle_score
                    particle.best_position = particle.position.copy()
                if particle_score > g_best_score:
                    g_best_score = particle_score
                    g_best_position = particle.position.copy()

            # Check for improvement
            if g_best_score > previous_g_best_score:
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            # Random Restart if no improvement for a while
            if no_improvement_counter >= MAX_ITER_WITHOUT_IMPROVEMENT:
                for particle in particles:
                    if np.random.rand() < 0.5:  # 50% chance to reinitialize a particle
                        particle.position = np.random.permutation(dimension)
                        particle.best_position = particle.position.copy()
                        particle.best_score = objective_function(particle.position)
                no_improvement_counter = 0  # Reset the counter

        # After the iterations, apply local search for each particle's best position
        for particle in particles:
            improved_position = local_search(particle.best_position)
            improved_score = objective_function(improved_position)
            if improved_score > particle.best_score:
                particle.best_position = improved_position
                particle.best_score = improved_score
            if improved_score > g_best_score:
                g_best_score = improved_score
                g_best_position = improved_position.copy()

        best_solutions.append(g_best_position)

    pool.close()
    return best_solutions

# Function to check if a solution is valid for the N-Queens problem
def is_solution_valid(position):
    n = len(position)
    # Check for row and column threats
    if len(set(position)) != n:
        return False
    # Check for diagonal threats
    for i in range(n):
        for j in range(i+1, n):
            if abs(position[i] - position[j]) == abs(i - j):  # Main diagonal threat
                return False
            if position[i] + i == position[j] + j:  # Counter diagonal threat
                return False
    return True

# Main execution
if __name__ == "__main__":
    # Input board size
    n = int(input("Enter the board size (n): "))
    while n < 4:
        print("Please enter a value for n that is greater than or equal to 4.")
        n = int(input("Enter the board size (n): "))

    # Determine the number of runs based on the board size
    # Determine the number of runs based on the board size
    if 4 <= n <= 17:
        num_runs = 50
    elif n == 18:
        num_runs = random.randint(50, 70)
    elif n == 20:
        num_runs = random.randint(70, 90)
    elif n == 24:
        num_runs = random.randint(90, 110)
    elif n == 30:
        num_runs = random.randint(110, 130)
    elif n == 36:
        num_runs = random.randint(130, 150)
    elif n == 48:
        num_runs = random.randint(150, 200)
    elif n == 52:
        num_runs = random.randint(200, 250)
    else:
    # Default case for other board sizes
        num_runs = random.randint(90, 110)


    # Input initial positions
    choice = input("Would you like to specify initial positions? (yes/no): ").lower()
    if choice == 'yes':
        initial_position = np.array(list(map(int, input(f"Enter the initial positions for the {n} queens column-wise: ").split())))
        while len(initial_position) != n:
            print(f"Please enter exactly {n} initial positions.")
            initial_position = np.array(list(map(int, input(f"Enter the initial positions for the {n} queens column-wise: ").split())))
    else:
        initial_position = np.arange(1, n+1)  # Using 1-based indexing
        np.random.shuffle(initial_position)

    # Choose parameter set or input custom parameters
    print("Choose a parameter set:")
    print("1: ", PARAMETER_SETS[1])
    print("2: ", PARAMETER_SETS[2])
    print("3: ", PARAMETER_SETS[3])
    print("4: Custom input")
    choice = int(input("Enter your choice (1/2/3/4): "))

    if choice in [1, 2, 3]:
        params = PARAMETER_SETS[choice]
        num_particles = params["num_particles"]
        num_iterations = params["num_iterations"]
        w = params["w"]
        c1 = params["c1"]
        c2 = params["c2"]
    else:
        num_particles = int(input("Enter number of particles: "))
        num_iterations = int(input("Enter number of iterations: "))
        w = float(input("Enter inertia weight (w): "))
        c1 = float(input("Enter cognitive parameter (c1): "))
        c2 = float(input("Enter social parameter (c2): "))

    # Run PSO and measure time taken
    start_time = time.time()
    solutions = PSO(num_particles, n, num_iterations, w, c1, c2, num_runs, initial_position)
    end_time = time.time()

    # Display results
    unique_solutions = set(tuple(sol) for sol in solutions if is_solution_valid(sol))
    for sol in unique_solutions:
        print(np.array(sol) + 1)
       
    print(f"\nNumber of unique solutions found: {len(unique_solutions)}")
    print(f"\nTotal time taken: {(end_time - start_time)/60:.2f} minutes")
