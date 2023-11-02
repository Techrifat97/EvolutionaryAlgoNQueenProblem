import numpy as np

class Bee:
    def __init__(self, n):
        self.position = np.random.permutation(n)
        self.fitness = self.evaluate_fitness()

    def evaluate_fitness(self):
        n = len(self.position)
        conflicts = 0
        for i in range(n):
            for j in range(i+1, n):
                if abs(self.position[i] - self.position[j]) == j - i:
                    conflicts += 1
        return -conflicts

    def modify_solution(self):
        idx1, idx2 = np.random.choice(len(self.position), 2, replace=False)
        self.position[idx1], self.position[idx2] = self.position[idx2], self.position[idx1]
        self.fitness = self.evaluate_fitness()

def bee_algorithm(n, num_scouts, num_foragers, max_iterations, max_trials):
    scouts = [Bee(n) for _ in range(num_scouts)]
    best_solution = max(scouts, key=lambda bee: bee.fitness)
    
    for iteration in range(max_iterations):
        # Employed Bee Phase
        for scout in scouts:
            scout.modify_solution()
            if scout.fitness > best_solution.fitness:
                best_solution = scout

        # Onlooker Bee Phase
        for _ in range(num_foragers):
            selected_scout = np.random.choice(scouts)
            selected_scout.modify_solution()
            if selected_scout.fitness > best_solution.fitness:
                best_solution = selected_scout

        # Scout Bee Phase
        for scout in scouts:
            trials = 0
            while trials < max_trials and scout.fitness < best_solution.fitness:
                scout.modify_solution()
                trials += 1
            if trials == max_trials:
                scout.position = np.random.permutation(n)
                scout.fitness = scout.evaluate_fitness()

    return best_solution.position, best_solution.fitness

if __name__ == "__main__":
    n = int(input("Enter the board size (n): "))
    num_scouts = int(input("Enter the number of scout bees: "))
    num_foragers = int(input("Enter the number of forager bees: "))
    max_iterations = int(input("Enter the maximum number of iterations: "))
    max_trials = int(input("Enter the maximum number of trials before a scout bee becomes a wanderer: "))

    best_position, best_fitness = bee_algorithm(n, num_scouts, num_foragers, max_iterations, max_trials)
    print("Best Solution:", best_position)
    print("Number of Conflicts:", -best_fitness)
