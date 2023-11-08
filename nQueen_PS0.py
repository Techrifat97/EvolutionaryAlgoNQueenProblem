"""@author: rifat_shaon"""
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

# Particle class represents a solution in the search space / population
class Particle:
    def __init__(self, initial_position):
        self.position = initial_position.copy()
        np.random.shuffle(self.position)
        self.velocity = np.zeros(len(self.position), dtype=int)
        self.best_position = np.copy(self.position)
        self.best_score = -float('inf')

# Objective function to evaluate the quality of a solution / fitness
def objective_function(position):
    n = len(position)
    # Count column conflicts
    column_conflicts = sum(np.bincount(position).clip(0, 1)) - n
    diagonal_conflicts = 0
    # Count diagonal conflicts
    for i in range(n):
        for j in range(i + 1, n):
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
        g_best_position = max(
            particles, key=lambda p: objective_function(p.position)).position.copy()
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
            scores = pool.map(objective_function, [
                              p.position for p in particles])
            for particle, score in zip(particles, scores):
                # Update particle velocity and position
                inertia = w * particle.velocity
                personal_attraction = c1 * np.random.random() * (particle.best_position -  particle.position)
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
                        particle.best_score = objective_function(
                            particle.position)
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
        for j in range(i + 1, n):
            if abs(position[i] - position[j]) == abs(i - j):  # Main diagonal threat
                return False
            if position[i] + i == position[j] + j:  # Counter diagonal threat
                return False
    return True


# Main execution
if __name__ == "__main__":
    # Input board size with validation
    while True:
        try:
            n = int(input("Enter the board size (n): "))
            if n < 4:
                print("Board size must be 4 or greater.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter an integer.")
# Determine the number of algorithm will runs based on the board size for generating multiple solutions
    while True:
        try:
            choice = input("Do you want to specify the number of times you want the algorithm to run?\nThe higher the number the chance of gettting multiple solution increases:(yes/no): ").strip().lower()
            if choice == 'yes':
                while True:
                    try:
                        num_runs = int(input("Enter the number of runs: "))
                        if num_runs <= 0:
                            print("Number of runs must be a positive integer.")
                        else:
                            break
                    except ValueError:
                        print("Invalid input. Please enter a positive integer.")
                break
            elif choice == 'no':
                if 4 <= n <= 17:
                    num_runs = 50
                elif n in [18, 20, 24, 30, 36, 48, 52]:
                    # If n matches predefined board sizes, use random within specific ranges
                    if n == 18:
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
                    num_runs = random.randint(90, 110)  # Default case for other board sizes
                break
            else:
                print("Invalid choice. Please enter 'yes' or 'no'.")
        except ValueError:
            print("Invalid input. Please enter a valid choice.")
            
    print(f"Number of times the algorithm will run to find multiple solutions is: {num_runs}")

    # Input initial positions with validation
    while True:
        choice = input("Would you like to specify initial positions? (yes/no): ").strip().lower()
        if choice not in ['yes', 'no']:
            print("Invalid input. Please enter 'yes' or 'no'.")
            continue

        if choice == 'yes':
            while True:
                input_positions = input(f"Enter the initial positions for the {n} queens column-wise (separated by space): ")
                try:
                    initial_position = np.array(list(map(int, input_positions.split())))
                    if len(initial_position) != n:
                        print(f"Please enter exactly {n} positions.")
                    elif not all(1 <= pos <= n for pos in initial_position):
                        print(f"Positions must be between 1 and {n}.")
                    else:
                        break
                except ValueError:
                    print("Invalid input. Please enter integers separated by spaces.")
            break
        else:
            initial_position = np.arange(n)
            np.random.shuffle(initial_position)
            break

    # Choose a parameter set or input custom parameters with validation
    while True:
        print("Choose a parameter set:")
        for key, value in PARAMETER_SETS.items():
            print(f"{key}: {value}")
        print("4: Custom input")
        choice = input("Enter your choice (1/2/3/4): ").strip()

        if choice in ['1', '2', '3']:
            params = PARAMETER_SETS[int(choice)]
            num_particles = params["num_particles"]
            num_iterations = params["num_iterations"]
            w = params["w"]
            c1 = params["c1"]
            c2 = params["c2"]
            break
        elif choice == '4':
            try:
                num_particles = int(input("Enter the number of particles: "))
                num_iterations = int(input("Enter the number of iterations: "))
                w = float(input("Enter inertia weight (w): "))
                c1 = float(input("Enter cognitive parameter (c1): "))
                c2 = float(input("Enter social parameter (c2): "))
                break
            except ValueError:
                print("Invalid input. Please enter the correct type of value.")
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")
    print("Wait for the solution......")
    # Run PSO and measure time taken
    start_time = time.time()
    solutions = PSO(num_particles, n, num_iterations, w, c1, c2, num_runs, initial_position)
    end_time = time.time()

    # Display results
    unique_solutions = {tuple(sol)
                        for sol in solutions if is_solution_valid(sol)}
    for sol in unique_solutions:
        print("Solution:", np.array(sol) + 1)  # Adjust for 1-based indexing
        
    print(f"\nNumber of unique solutions found: {len(unique_solutions)}")
    print(f"Total time taken: {(end_time - start_time):.2f} seconds")
