#SHAYAAN HASNAIN AHMAD
#20I-0647
#SECTION A
# AI - ASSIGNMENT 2

import random
from typing import List, Tuple

# DEFINING THE NUMBER OF COURSES C1-C5, NUMBER OF HALLS H1,H2
# THE TIMESLOTS AVAILABLE T1,T2,T3 AND THE MAX AMOUNT OF TIME A HALL COULD BE OCCUPIED THAT IS 6HRS

NUM_TIMESLOTS = 3
MAX_HALL_HOURS = 6

NUM_COURSES = 5
NUM_HALLS = 2

# DEFINING THE COURSE CONFLICTS IN THE FROM (COURSE,COURSE,NUMBER OF STUDENTS)
# E.G (C1,C2,10) -> THIS MEANS C1 AND C2 HAVE 10 CONFLICTING STUDENTS
conflicts = [(1, 2, 10), (1, 4, 5), (2, 5, 7), (3, 4, 12), (4, 5, 8)]

# HOW THE SOLUTION WOULD BE RETURNED 
# COURSE, TIME SLOT, HALL NUMBER 
Solution = List[Tuple[int, int, int]]  # course, time slot, hall


def generate_random_solution() -> Solution:

   # This function randomly assignes each course to a timeslot and a hall.
    solution = []
    for i in range(NUM_COURSES):
        course = i + 1
        timeslot = random.randint(1, NUM_TIMESLOTS)
        hall = random.randint(1, NUM_HALLS)
        solution.append((course, timeslot, hall))
    return solution


def calculate_conflicts(solution: Solution) -> int:
    """Calculate the number of conflicts in a solution"""
    #This function calculates the number of conflicts in a given solution.
    #checks conflicts list, which specifies which courses cannot be scheduled at the same time and in the same hall.
    #If any two conflicting courses are scheduled at the same time and in the same hall in the given solution, then a conflict is counted. 
    conflicts_count = 0  #to count conflicts
    for c1, c2, _ in conflicts:
        if any(s1[0] == c1 and s2[0] == c2 and s1[1] == s2[1] and s1[2] != s2[2] for s1 in solution for s2 in solution):
            conflicts_count += 1
    return conflicts_count


def calculate_penalty(solution: Solution) -> int:
    # Finds the penalty for a solution
    # If a hall is used for more than MAX_HALL_HOURS hours, then a penalty of 10 is incurred.
    # If two conflicting courses are scheduled in the same timeslot but different halls, then a penalty of 100 is incurred
    penalty = 0
    hall_hours = [0] * NUM_HALLS
    for course, timeslot, hall in solution:
        hall_hours[hall - 1] += 1
        if hall_hours[hall - 1] > MAX_HALL_HOURS:
            penalty += 10 #CHANGE PENALTY ACCORDING TO CHOICE -> for exceeding time
        for c1, c2, _ in conflicts:
            if course == c1 and any(s[0] == c2 and s[1] == timeslot and s[2] != hall for s in solution):
                penalty += 100 #CHANGE PENALTY ACCORDING TO CHOICE -> for same timeslot
    return penalty


def calculate_fitness(solution: Solution) -> int:
  #The fitness function should take a solution as input and output a score indicating how good the solution
  #It takes in a Solution and returns an integer representing the fitness value of the solution.
  #The fitness value is calculated as the sum of two components: the number of conflicts in the solution and the penalty for the solution.
  #The fitness value is the sum of the number of conflicts and the penalty, so a lower fitness value indicates a better solution. 
    """Calculate the fitness of a solution"""
    conflicts = calculate_conflicts(solution)
    penalty = calculate_penalty(solution)
    return conflicts + penalty


def generate_population(size: int) -> List[Solution]:
    #Generates a population of random solutions
    return [generate_random_solution() for _ in range(size)]


def tournament_selection(population: List[Solution], size: int) -> Solution:
    #Select the best solution from a random sample of the population
    #This function performs tournament selection on a given population. 
    #It does this by selecting a random sample of size solutions from the population, and then returning the solution with the lowest fitness
    sample = random.sample(population, size)
    return min(sample, key=calculate_fitness)


def single_point_crossover(parent1: Solution, parent2: Solution, probability: float) -> Tuple[Solution, Solution]:
    #Perform single-point crossover on two parent solutions
    #selecting a random crossover point between the first and last courses, and then swapping the courses after that point between the two parents to create two children. T
    if random.random() > probability:
        return parent1, parent2
    crossover_point = random.randint(1, NUM_COURSES - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutation(solution: Solution, probability: float) -> Solution:
    #The mutation function is responsible for mutating a single solution in the population. 
    #It randomly changes the timeslot and hall of a randomly selected course in the given solution with a certain probability. 
    if random.random() > probability:
        return solution
    mutated_solution = solution[:]
    course = random.randint(1, NUM_COURSES)
    timeslot = random.randint(1, NUM_TIMESLOTS)
    hall = random.randint(1, NUM_HALLS)
    mutated_solution[course - 1] = (course, timeslot, hall)
    return mutated_solution


def genetic_algorithm(population_size: int, tournament_size: int, crossover_probability: float, mutation_probability: float,
                      num_generations: int) -> Tuple[Solution, int]:
    #Solve the scheduling problem using a genetic algorithm
    #The algorithm terminates after num_generations generations, and the best solution found is returned.
    #Does not need derivative information
    #Provides ans that improve over time
    #Larger search space
    population = generate_population(population_size)
    for generation in range(num_generations):
        next_population = []
        for _ in range(population_size):
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)
            child1, child2 = single_point_crossover(parent1, parent2, crossover_probability)
            mutated_child1 = mutation(child1, mutation_probability)
            mutated_child2 = mutation(child2, mutation_probability)
            next_population.extend([mutated_child1, mutated_child2])
        population = [tournament_selection(next_population, tournament_size) for _ in range(population_size)]
    best_solution = min(population, key=calculate_fitness)
    best_fitness = calculate_fitness(best_solution)
    return best_solution, best_fitness


if __name__ == '__main__':
    best_solution, best_fitness = genetic_algorithm(population_size=100, tournament_size=5, crossover_probability=0.8,
                                                    mutation_probability=0.1, num_generations=100)
    
    print("ANS IS IN THE FORM OF COURSE NUMBER,TIME SLOT, HALL NUMBER")
    print("MOST OPTIMAL/BEST SOLUTION:", best_solution)
    print("BEST FITNESS VALUE:", best_fitness)
