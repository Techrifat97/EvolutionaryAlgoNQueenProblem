import random
import time
BEE_ALGORITHM_PARAMETER_SETS = {
    1: {"num_scouts": 50, "num_best_sites": 1, "num_bees_best_sites": 50, "num_other_sites": 1, "num_bees_other_sites": 50, "max_iterations": 500, "ngh": 50, "stlim": 50},
    2: {"num_scouts": 100, "num_best_sites": 2, "num_bees_best_sites": 100, "num_other_sites": 2, "num_bees_other_sites": 100, "max_iterations": 600, "ngh": 100, "stlim": 100},
    3: {"num_scouts": 150, "num_best_sites": 3, "num_bees_best_sites": 150, "num_other_sites": 3, "num_bees_other_sites": 150, "max_iterations": 700, "ngh": 150, "stlim": 150}
}
def initialize_board(n):
    return [[str(i) for i in range(1, n + 1)] for _ in range(n)]

def cost(positions):
    """Calculate the number of pairs of queens that are attacking each other"""
    n = len(positions)
    attack = 0
    for i in range(n):
        for j in range(i + 1, n):
            if positions[i] == positions[j] or abs(positions[i] - positions[j]) == j - i:
                attack += 1
    return attack

def heuristic_initial_positions(n):
    """Generate a semi-random initial position based on a heuristic"""
    positions = list(range(n))
    random.shuffle(positions)
    return positions

def get_user_input(n):
    """Get user input for initial positions or generate random positions"""
    choice = input("Do you want to input initial positions or use random initial positions? (input/random): ").strip().lower()
    if choice == 'input':
        while True:
            try:
                initial_positions = input(f"Enter the initial positions of the queens for a {n}x{n} board, separated by spaces (1-{n}): ")
                positions = list(map(int, initial_positions.split()))
                if len(positions) != n or not all(1 <= pos <= n for pos in positions):
                    raise ValueError
                return positions
            except ValueError:
                print(f"Invalid input. Please enter {n} positions from 1 to {n}.")
    elif choice == 'random':
        return heuristic_initial_positions(n)
    else:
        print("Invalid choice. Please enter 'input' or 'random'.")
        return get_user_input(n)

def local_search(solution, n):
    """Perform local search to improve a given solution"""
    for col in range(n):
        min_conflicts = n
        best_row = solution[col]
        for row in range(n):
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

def site_abandonment(solutions, costs, threshold):
    """Abandon sites that have not improved past a certain threshold"""
    return [(solution, cost) for solution, cost in zip(solutions, costs) if cost <= threshold]


def neighbourhood_shrinking(solution, n, ngh):
    """Reduce the neighborhood size to intensify the search"""
    col = random.randint(0, n - 1)
    best_row = solution[col]
    min_conflicts = cost(solution)
    for _ in range(ngh):
        row_change = random.randint(-1, 1)
        new_row = (solution[col] + row_change) % n
        solution[col] = new_row
        conflicts = cost(solution)
        if conflicts < min_conflicts:
            min_conflicts = conflicts
            best_row = new_row
    solution[col] = best_row
    return solution

def global_search(num_scouts, n):
    """Perform a global search to escape local optima"""
    return [heuristic_initial_positions(n) for _ in range(num_scouts)]

def bees_algorithm(n, num_scouts, num_best_sites, num_bees_best_sites, num_other_sites, num_bees_other_sites, max_iterations, ngh, stlim):
    """The Bees Algorithm for solving the N-Queens problem"""
    scout_solutions = [heuristic_initial_positions(n) for _ in range(num_scouts)]
    best_solution = None
    best_cost = float('inf')
    some_interval = 10
    some_threshold = 2  # This threshold could be a parameter or dynamically adjusted
    no_improvement_runs = 0
    max_no_improvement_runs = stlim

    for iteration in range(max_iterations):
        # Evaluate all scout solutions
        costs = [cost(solution) for solution in scout_solutions]
        sorted_solutions = sorted(zip(scout_solutions, costs), key=lambda x: x[1])

        # Select best sites and perform local search
        best_sites = sorted_solutions[:num_best_sites]
        for i, (solution, solution_cost) in enumerate(best_sites):
            for _ in range(num_bees_best_sites):
                improved_solution = local_search(solution, n)
                improved_cost = cost(improved_solution)
                if improved_cost < solution_cost:
                    best_sites[i] = (improved_solution, improved_cost)
                    solution_cost = improved_cost

        # Abandon sites that have not improved past the threshold
        best_sites = site_abandonment([site for site, cost in best_sites], [cost for site, cost in best_sites], some_threshold)

        # Check for early termination
        if best_sites:
            current_best_solution, current_best_cost = min(best_sites, key=lambda x: x[1])
            if current_best_cost < best_cost:
                best_solution, best_cost = current_best_solution, current_best_cost
                no_improvement_runs = 0
            else:
                no_improvement_runs += 1
                if no_improvement_runs >= max_no_improvement_runs:
                    break
        else:
            # If all sites are abandoned, then regenerate the scout solutions
            scout_solutions = global_search(num_scouts, n)
            continue

        # Dynamic Neighborhood Shrinkage
        if iteration % some_interval == 0 and iteration > 0:
            for i in range(len(scout_solutions)):
                scout_solutions[i] = neighbourhood_shrinking(scout_solutions[i], n, ngh)

        # Replace the abandoned sites with new random scouts
        while len(best_sites) < num_best_sites:
            new_scout = heuristic_initial_positions(n)
            new_cost = cost(new_scout)
            best_sites.append((new_scout, new_cost))

        # Update the scout solutions with the best sites
        scout_solutions = [solution for solution, cost in best_sites]

    return best_solution


def is_solution_valid(position):
    n = len(position)
    if len(set(position)) != n:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if abs(position[i] - position[j]) == abs(i - j):
                return False
    return True

# Example usage
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
    num_runs = 50  # Default value
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
