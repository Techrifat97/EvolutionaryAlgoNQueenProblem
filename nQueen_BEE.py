import random
import time

# Define parameter sets for the Bee Algorithm with different levels of intensity and search space.
BEE_ALGORITHM_PARAMETER_SETS = {
    # Parameter set 1 for smaller boards or quicker runs
    1: {"num_scouts": 50, "num_best_sites": 1, "num_bees_best_sites": 50, "num_other_sites": 1, "num_bees_other_sites": 50, "max_iterations": 500, "ngh": 50, "stlim": 50},
    # Parameter set 2 for medium-sized boards or moderate runs
    2: {"num_scouts": 100, "num_best_sites": 2, "num_bees_best_sites": 100, "num_other_sites": 2, "num_bees_other_sites": 100, "max_iterations": 600, "ngh": 100, "stlim": 100},
    # Parameter set 3 for larger boards or more intensive runs
    3: {"num_scouts": 150, "num_best_sites": 3, "num_bees_best_sites": 150, "num_other_sites": 3, "num_bees_other_sites": 150, "max_iterations": 700, "ngh": 150, "stlim": 150}
}

# Function to calculate the cost (number of attacking pairs of queens)
def cost(positions):
    n = len(positions)
    attack = 0
    for i in range(n):
        for j in range(i + 1, n):
            if positions[i] == positions[j] or abs(positions[i] - positions[j]) == j - i:
                attack += 1
    return attack

# Function to generate a semi-random initial position based on a heuristic
def heuristic_initial_positions(n):
    positions = list(range(n))
    random.shuffle(positions)
    return positions

# Function to get user input for initial positions or generate random positions
def get_user_input(n):
    choice = input("Do you want to input initial positions or use random initial positions? (yes/no): ").strip().lower()
    if choice == 'yes':
        while True:
            try:
                initial_positions = input(f"Enter the initial positions of the queens for a {n}x{n} board, separated by spaces (1-{n}): ")
                positions = list(map(int, initial_positions.split()))
                if len(positions) != n or not all(1 <= pos <= n for pos in positions):
                    raise ValueError
                return positions  # Return the positions without checking for threats
            except ValueError:
                print(f"Invalid input. Please enter {n} numbers from 1 to {n}, representing the row positions of the queens.")
    elif choice == 'no':
        return heuristic_initial_positions(n)
    else:
        print("Invalid choice. Please enter 'input' or 'random'.")
        return get_user_input(n)

# Function to perform local search to improve a given solution
def local_search(solution, n, ngh):
    for col in range(n):
        min_conflicts = n
        best_row = solution[col]
        # Explore the neighborhood of the current position within the range defined by ngh
        for row in range(max(0, col - ngh), min(n, col + ngh + 1)):
            if row != solution[col]:
                solution[col] = row
                conflicts = cost(solution)
                if conflicts < min_conflicts:
                    min_conflicts = conflicts
                    best_row = row
        solution[col] = best_row
        if min_conflicts == 0:
            break  # Stop if we find a position with no conflicts
    return solution


# Function to abandon sites that have not improved past a certain threshold
def site_abandonment(solutions, costs, threshold):
     return [(solution, cost) for solution, cost in zip(solutions, costs) if cost <= threshold]

# Function to perform a global search to escape local optima
def global_search(num_scouts, n):
    return [heuristic_initial_positions(n) for _ in range(num_scouts)]

# Main function implementing the Bees Algorithm for solving the N-Queens problem
def bees_algorithm(n, num_scouts, num_best_sites, num_bees_best_sites, num_other_sites, num_bees_other_sites, max_iterations, ngh, stlim):
    # Initialize scout solutions
    scout_solutions = [heuristic_initial_positions(n) for _ in range(num_scouts)]
    best_solution = None
    best_cost = float('inf')
    some_interval = 10  # Interval for neighborhood shrinking
    some_threshold = 2  # Threshold for site abandonment
    no_improvement_runs = 0
    max_no_improvement_runs = stlim  # Maximum runs without improvement before stopping

    for iteration in range(max_iterations):
        # Evaluate all scout solutions
        costs = [cost(solution) for solution in scout_solutions]
        sorted_solutions = sorted(zip(scout_solutions, costs), key=lambda x: x[1])

        # Select best sites and perform local search
        best_sites = sorted_solutions[:num_best_sites]
        for i, (solution, solution_cost) in enumerate(best_sites):
            for _ in range(num_bees_best_sites):
                improved_solution = local_search(solution, n, ngh)
                improved_cost = cost(improved_solution)
                if improved_cost < solution_cost:
                    best_sites[i] = (improved_solution, improved_cost)
                    solution_cost = improved_cost

        # Select other sites and perform local search
        other_sites = sorted_solutions[num_best_sites:num_best_sites+num_other_sites]
        for i, (solution, solution_cost) in enumerate(other_sites):
            for _ in range(num_bees_other_sites):
                improved_solution = local_search(solution, n, ngh)
                improved_cost = cost(improved_solution)
                if improved_cost < solution_cost:
                    other_sites[i] = (improved_solution, improved_cost)
                    solution_cost = improved_cost

        # Combine best and other sites after local search
        combined_sites = best_sites + other_sites
        combined_solutions = [site for site, _ in combined_sites]
        combined_costs = [cost for _, cost in combined_sites]

        # Abandon sites that have not improved past the threshold
        combined_sites = site_abandonment(combined_solutions, combined_costs, some_threshold)
        # Check for early termination if no improvement is observed
        if combined_sites:
            current_best_solution, current_best_cost = min(combined_sites, key=lambda x: x[1])
            if current_best_cost < best_cost:
                best_solution, best_cost = current_best_solution, current_best_cost
                no_improvement_runs = 0
            else:
                no_improvement_runs += 1
                if no_improvement_runs >= max_no_improvement_runs:
                    break
        else:
            # If all sites are abandoned, regenerate the scout solutions
            scout_solutions = global_search(num_scouts, n)
            continue

        # Perform neighborhood shrinking at certain intervals
        if iteration % some_interval == 0 and iteration > 0:
            for i in range(len(scout_solutions)):
                scout_solutions[i] = local_search(scout_solutions[i], n, ngh)

        # Replace abandoned sites with new random scouts
        while len(combined_sites) < num_best_sites + num_other_sites:
            new_scout = heuristic_initial_positions(n)
            new_cost = cost(new_scout)
            combined_sites.append((new_scout, new_cost))

        # Update scout solutions with the combined sites
        scout_solutions = [solution for solution, cost in combined_sites]

    return best_solution

# Function to check if a solution is valid (no queens are attacking each other)
def is_solution_valid(position):
    n = len(position)
    if len(set(position)) != n:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if abs(position[i] - position[j]) == abs(i - j):
                return False
    return True

# Example usage of the algorithm
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

    # Determine the number of algorithm runs
    num_runs = 50  # Default number of runs
    choice = input("Do you want to specify the number of times you want the algorithm to run?\n"
                   "The higher the number, the chance of getting multiple solutions increases (yes/no): ").strip().lower()
    if choice == 'yes':
        num_runs = int(input("Enter the number of runs: "))
        while num_runs <= 0:
            print("Number of runs must be a positive integer.")
            num_runs = int(input("Enter the number of runs: "))

    # Get initial positions from the user or generate random initial positions
    initial_positions = get_user_input(n)
    
    # Parameter set selection or custom parameter input
    params = None
    while True:
        print("Choose a parameter set:")
        for key, value in BEE_ALGORITHM_PARAMETER_SETS.items():
            print(f"{key}: {value}")
        print("4: Custom input")
        choice = input("Enter your choice (1/2/3/4): ").strip()
        
        if choice in ['1', '2', '3']:
            params = BEE_ALGORITHM_PARAMETER_SETS[int(choice)]
            break
        elif choice == '4':
            try:
                params = {
                    'num_scouts': int(input("Enter the number of scouts: ")),
                    'num_best_sites': int(input("Enter the number of best sites: ")),
                    'num_bees_best_sites': int(input("Enter the number of bees per best site: ")),
                    'num_other_sites': int(input("Enter the number of other sites: ")),
                    'num_bees_other_sites': int(input("Enter the number of bees per other site: ")),
                    'max_iterations': int(input("Enter the maximum number of iterations: ")),
                    'ngh': int(input("Enter the neighborhood size (ngh): ")),
                    'stlim': int(input("Enter the maximum number of iterations with no improvement (stlim): "))
                }
                break
            except ValueError:
                print("Invalid input. Please enter positive integers.")
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

    # Run the Bee Algorithm
    solutions = []
    start_time = time.time()  # Start timing
    print("Running the Bee Algorithm...")
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        best_solution = bees_algorithm(n, **params)
        if best_solution and is_solution_valid(best_solution):
            solutions.append(best_solution)

    end_time = time.time()  # End timing
    time_spent = end_time - start_time  # Calculate time spent

    # Display results
    if solutions:
        print("\nSolutions found:")
        for i, solution in enumerate(solutions):
            adjusted_solution = [pos + 1 for pos in solution]
            print(f"Solution {i + 1} (positions): {adjusted_solution}")
        print(f"\nTime spent: {time_spent:.2f} seconds")
    else:
        print("\nNo valid solutions found.")
        print(f"\nTime spent: {time_spent:.2f} seconds")
