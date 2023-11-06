#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 5 22:19:04 2023

@author: MirzaTamzid
"""
import random
from itertools import accumulate
import time

class ACO:
    def __init__(self, n, num_ants, evaporation_rate, alpha, beta, iterations):
        self.n = n
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.pheromone = [[1 for _ in range(n)] for _ in range(n)]

    def run(self):
        best_solution = None
        best_fitness = float('-inf')

        for _ in range(self.iterations):
            solutions = self.construct_solutions()
            self.update_pheromone(solutions)

            for solution in solutions:
                fitness = self.fitness(solution)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = solution

        return best_solution

    def construct_solutions(self):
        solutions = []
        for _ in range(self.num_ants):
            solution = []
            for _ in range(self.n):
                next_queen = self.select_next_queen(solution)
                solution.append(next_queen)
            solutions.append(solution)
        return solutions

    def select_next_queen(self, solution):
        probabilities = self.calculate_probabilities(solution)
        return self.roulette_wheel_selection(probabilities)

    def calculate_probabilities(self, solution):
        pheromone = [self.pheromone[len(solution)][i] for i in range(self.n)]
        heuristic = [1 / (1 + self.conflicts(solution, i)) for i in range(self.n)]
        total = sum(p ** self.alpha * h ** self.beta for p, h in zip(pheromone, heuristic))

        probabilities = [(pheromone[i] ** self.alpha * heuristic[i] ** self.beta) / total for i in range(self.n)]
        return probabilities

    def conflicts(self, solution, position):
        conflicts = 0
        for i, q in enumerate(solution):
            if q == position or abs(len(solution) - i) == abs(q - position):
                conflicts += 1
        return conflicts

    def roulette_wheel_selection(self, probabilities):
        cumulative_prob = list(accumulate(probabilities))
        r = random.random()
        for i, prob in enumerate(cumulative_prob):
            if r <= prob:
                return i
        return len(probabilities) - 1

    def update_pheromone(self, solutions):
        for i in range(self.n):
            for j in range(self.n):
                self.pheromone[i][j] *= (1 - self.evaporation_rate)

        for solution in solutions:
            fitness = self.fitness(solution)
            for i, q in enumerate(solution):
                if i < self.n and q < self.n:
                    self.pheromone[i][q] += 1 / (fitness + 1e-5)

    def fitness(self, solution):
        total_conflicts = sum(self.conflicts(solution, solution[i]) for i in range(len(solution)))
        non_attacking_pairs = (self.n * (self.n - 1) // 2) - total_conflicts
        return non_attacking_pairs

def main():
    n = int(input("Enter the board size (n): "))
    _ = input("Enter the initial positions of the queens: ")
    num_ants = int(input("Enter the number of ants: "))
    evaporation_rate = float(input("Enter the pheromone evaporation rate: "))
    alpha = float(input("Enter alpha (influence of pheromone): "))
    beta = float(input("Enter beta (influence of heuristic information): "))
    iterations = int(input("Enter the number of iterations: "))
    runs = int(input("How many times should the algorithm run? "))

    best_solution_over_runs = None
    best_fitness_over_runs = float('-inf')

    start_time = time.time()  # Start the timer

    for _ in range(runs):
        aco = ACO(n, num_ants, evaporation_rate, alpha, beta, iterations)
        solution = aco.run()
        fitness = aco.fitness(solution)
        if fitness > best_fitness_over_runs:
            best_fitness_over_runs = fitness
            best_solution_over_runs = solution

    end_time = time.time()  # End the timer

    if best_solution_over_runs is not None:
        print("Best solution found:", [pos + 1 for pos in best_solution_over_runs])  # Adjusting back to 1-indexed for output
    else:
        print("No solution found.")

    print("Runtime of the algorithm: {:.4f} seconds".format(end_time - start_time))

if __name__ == "__main__":
    main()
