import math
import random

from models.data_init import DataInit

class GeneticAlgorithm:
    def __init__(self, data_init: DataInit):
        self.data_init: DataInit = data_init
        self.func_range = data_init.b_interval - data_init.a_interval
        self.system_resolution = None

    # Función que retorna la resolución del sistema y el número de bits a usar. Recibe el rango de la función y el valor de la resolución del problema.
    def system_resolution_and_bits(self, problem_resolution: float = None, func_range: float = None) -> tuple[float, int]:
        
        if func_range is None:
            func_range = self.func_range
        
        if problem_resolution is None:
            problem_resolution = self.data_init.problem_resolution

        # Encontrar los puntos del problema
        problem_points = math.ceil(func_range / problem_resolution) + 1
        bits = math.ceil(math.log2(problem_points))

        system_points = 2**bits
        system_resolution = func_range / (system_points - 1)    
        
        self.system_resolution = system_resolution

        return system_resolution, bits
    
    # Crear a un individuo
    def __create_individual(self, bytes_length: int) -> list[int]:
        chromosome = [random.randint(0, 1) for i in range(bytes_length)] 
        return chromosome 

    # Crear la población inicial
    def generate_initial_population(self, bytes_length: int, initial_population: int = None) -> list:
        if initial_population is None:
            initial_population = self.data_init.initial_population
        return [self.__create_individual(bytes_length) for i in range(initial_population)]
    
    # Funión para convertir cadena binaria en entero
    def __binary_to_int(self, value: list[int]) -> int:
        return int("".join(map(str, value)), 2)
    
    # Función para hallar el fenotipo
    def calculate_phenotype(self, individual: list[int]) -> float:
        if self.system_resolution is None:
            raise ValueError("The solution to the system has not been found")
        
        index = self.__binary_to_int(individual)
        return self.data_init.a_interval + (index * self.system_resolution)
 
    # Evaluar la población
    def evaluate_population(self, population: list[list[int]]) -> list[tuple]:
        """
        Recibe la población completa y devuelve una lista de tuplas:
        [(individuo, indice, fitness), ...]
        """
        evaluated_population = []

        for individual in population:
            # 1. Obtener el valor real X
            x = self.calculate_phenotype(individual)
            
            # 2. Evaluar en la función objetivo f(x)
            # Usamos la función que pasaste en data_init
            fitness = self.data_init.func(x)
            
            # Guardamos todo para tener trazabilidad
            evaluated_population.append((individual, x, fitness))
            
        return evaluated_population
    
    # Crear parejas
    def generate_pairs(self, population: list[list[int]], crossover_threshold: float = None)-> list[tuple]:
        
        if crossover_threshold is None:
            crossover_threshold = self.data_init.crossover_threshold
        
        pairs = []
        n = len(population) 
        

        for i in range(n): # Para cada individuo de la población en i
            for j in range(n): # para cadad individuo de la población en j
                if i != j:
                    p = random.random()

                    if p <= crossover_threshold:
                        pair = (population[i], population[j])
                        pairs.append(pair)

        return pairs
    
    # Cruza
    def __one_point_crossover_random(self, parent1: list[int], parent2: list[int]) -> tuple:
        
        if len(parent1) <2: return parent1, parent2 

        point = random.randint(1, len(parent1)-1)

        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]

        return child1, child2
    
    def cross(self, pairs: list[tuple]) -> list[list[int]]:
        childs = [] 
        
        for parent1, parent2 in pairs:
            child1, child2 = self.__one_point_crossover_random(parent1, parent2)
            
            childs.append(child1)
            childs.append(child2)
            
        return childs
    
    # Mutación
    def mutate(self, childs: list[list[int]], individual_mutation_threshold = None, gene_mutation_threshold = None) -> list[list[int]]:
        
        if individual_mutation_threshold is None:
            individual_mutation_threshold = self.data_init.individual_mutation_threshold
        
        if gene_mutation_threshold is None:
            gene_mutation_threshold = self.data_init.gene_mutation_threshold

        for individual in childs:
            p_ind = random.random()

            if p_ind <= individual_mutation_threshold:
                for j in range(len(individual)):
                    p_gen = random.random()
                    
                    if p_gen <= gene_mutation_threshold:
                        if individual[j] == 0:
                            individual[j] = 1
                        else:
                            individual[j] = 0
        
        return childs
    
    # Poda:
    def prune(self, population: list, childs: list[list[int]], maximize: bool = True)-> tuple[list[list[int]], list[tuple]]:
        max_pup = self.data_init.max_population
        
        full_population = population + childs

        evaluated = self.evaluate_population(full_population)

        evaluated.sort(key=lambda x: x[2], reverse=maximize)

        next_generation = []

        best_individual = evaluated[0]
        next_generation.append(best_individual[0]) 

        evaluated.pop(0)

        # Dividir en tres partes
        length = len(evaluated)

        n_part = length // 3

        part1 = evaluated[: n_part]
        part2 = evaluated[n_part : n_part * 2]
        part3 = evaluated[n_part * 2 :]

        while len(next_generation) < max_pup:
            
            block = random.randint(1, 3)

            if block == 1:
                index = random.randint(0, len(part1)-1)
                selected_tuple = part1[index]
                next_generation.append(selected_tuple[0])
                part1.pop(index)
            
            if block == 2:
                index = random.randint(0, len(part2)-1)
                selected_tuple = part2[index]
                next_generation.append(selected_tuple[0])
                part2.pop(index)
            
            if block == 3:
                index = random.randint(0, len(part3)-1)
                selected_tuple = part3[index]
                next_generation.append(selected_tuple[0])
                part3.pop(index)

        return next_generation







    