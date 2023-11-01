import numpy as np
from multiprocessing import Pool, cpu_count
import time
import random

PARAMETER_SETS = {
    1: {"num_particles": 100, "num_iterations": 500, "w": 0.9, "c1": 1.2, "c2": 2.2},
    2: {"num_particles": 150, "num_iterations": 600, "w": 0.8, "c1": 1.3, "c2": 2.1},
    3: {"num_particles": 200, "num_iterations": 700, "w": 0.7, "c1": 1.4, "c2": 2.0}
}


class Particle:
    def __init__(self, initial_position):
        self.position = initial_position.copy()
        np.random.shuffle(self.position)
        self.velocity = np.zeros(len(self.position), dtype=int)
        self.best_position = np.copy(self.position)
        self.best_score = -float('inf')

def objective_function(position):
    n = len(position)
    column_conflicts = sum(np.bincount(position).clip(0, 1)) - n
    diagonal_conflicts = 0

    for i in range(n):
        for j in range(i+1, n):
            if position[i] - position[j] == i - j or position[j] - position[i] == i - j:
                diagonal_conflicts += 1

    return -(column_conflicts + diagonal_conflicts)


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

def PSO(num_particles, dimension, num_iterations, w, c1, c2, num_runs, initial_position):
    best_solutions = []
    pool = Pool(processes=cpu_count())  # Create a multiprocessing Pool

    for run in range(num_runs):
        particles = [Particle(initial_position) for _ in range(num_particles)]
        scores = pool.map(objective_function, [p.position for p in particles])
        g_best_position = max(particles, key=lambda p: objective_function(p.position)).position.copy()
        g_best_score = objective_function(g_best_position)

        for particle, score in zip(particles, scores):
            particle.best_score = score
            particle.best_position = particle.position.copy()

        for iteration in range(num_iterations):
            scores = pool.map(objective_function, [p.position for p in particles])
            for particle, score in zip(particles, scores):
                inertia = w * particle.velocity
                personal_attraction = c1 * np.random.random() * (particle.best_position - particle.position)
                social_attraction = c2 * np.random.random() * (g_best_position - particle.position)
                particle.velocity = inertia + personal_attraction + social_attraction
                swap_idx1 = int(abs(particle.velocity[0]) % dimension)
                swap_idx2 = int(abs(particle.velocity[1]) % dimension)
                particle.position[swap_idx1], particle.position[swap_idx2] = particle.position[swap_idx2], particle.position[swap_idx1]
                particle.position = particle.position % dimension

                particle_score = objective_function(particle.position)
                if particle_score > particle.best_score:
                    particle.best_score = particle_score
                    particle.best_position = particle.position.copy()
                if particle_score > g_best_score:
                    g_best_score = particle_score
                    g_best_position = particle.position.copy()

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

def is_solution_valid(position):
    """Check if the given solution is valid for the N-Queens problem."""
    n = len(position)

    # Check for row threats
    if len(set(position)) != n:
        return False

    # Check for column threats (implicitly handled by the above check)

    # Check for diagonal threats
    for i in range(n):
        for j in range(i+1, n):
            if abs(position[i] - position[j]) == abs(i - j):  # Main diagonal threat
                return False
            if position[i] + i == position[j] + j:  # Counter diagonal threat
                return False

    return True

if __name__ == "__main__":
    n = int(input("Enter the board size (n): "))
    while n < 4:
        print("Please enter a value for n that is greater than or equal to 4.")
        n = int(input("Enter the board size (n): "))

    # Determine the number of runs based on the board size
    if n <= 8:
        num_runs = random.randint(30, 50)
    elif n <= 12:
        num_runs = random.randint(50, 70)
    elif n <= 16:
        num_runs = random.randint(70, 90)
    else:
        num_runs = random.randint(90, 110)
    
    #print(f"Number of runs set to: {num_runs}")

    choice = input("Would you like to specify initial positions? (yes/no): ").lower()
    if choice == 'yes':
        initial_position = np.array(list(map(int, input(f"Enter the initial positions for the {n} queens column-wise: ").split())))
        while len(initial_position) != n:
            print(f"Please enter exactly {n} initial positions.")
            initial_position = np.array(list(map(int, input(f"Enter the initial positions for the {n} queens column-wise: ").split())))
    else:
        initial_position = np.arange(1, n+1)  # Using 1-based indexing
        np.random.shuffle(initial_position)

    # Allow the user to choose one of the predefined parameter sets or input their own
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

    start_time = time.time()
    solutions = PSO(num_particles, n, num_iterations, w, c1, c2, num_runs, initial_position)
    end_time = time.time()

    unique_solutions = set(tuple(sol) for sol in solutions if is_solution_valid(sol))
    for sol in unique_solutions:
        print(np.array(sol) + 1)
       
    print(f"\nNumber of unique solutions found: {len(unique_solutions)}")
    print(f"\nTotal time taken: {(end_time - start_time)/60:.2f} minutes")
