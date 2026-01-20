from GeneticAlgorithm import GeneticAlgorithm
from models.data_init import DataInit
import numpy as np


def func(x):
    return x * np.cos(7*x) + np.sin(3*x)

data_init = DataInit(
    problem_resolution=0.25,
    a_interval=20,
    b_interval=22,
    initial_population=10,
    max_population=30,
    crossover_threshold=0.2,
    individual_mutation_threshold=0.2,
    gene_mutation_threshold=0.25,
    iterations=100,
    func=func
)

genetic = GeneticAlgorithm(data_init=data_init)

system_res, bytes_length = genetic.system_resolution_and_bits()
print(f"Solucion del sistema: {system_res}, \nNúmero de bytes: {bytes_length} \n")

print("Población inicial:\n")
population = genetic.generate_initial_population(bytes_length)
for ind in population:
    print(ind)

print(f"--- INICIO ---")
print(f"Población Inicial: {len(population)}")

# Ciclo de Generaciones
for i in range(data_init.iterations):
    
    # 2. Generar Parejas 
    pairs = genetic.generate_pairs(population)
    
    # 3. Cruzar Parejas 
    offspring = genetic.cross(pairs)

    # 4. Mutar
    childs = genetic.mutate(offspring)
    
    # 4. PODA 
    population = genetic.prune(population, childs)
    
    # --- Reporte de la generación ---
    # Evaluamos solo al mejor para mostrar progreso
    best_ind = genetic.evaluate_population([population[0]])[0] # El 0 es el mejor porque prune ya ordenó
    print(f"Generación {i}: Mejor Fitness = {best_ind[2]:.5f} | Tamaño Pob: {len(population)}")

print("\n--- FIN DEL ALGORITMO ---")
print(f"Mejor individuo final: {population[0]}")
print(f"\nFenotipo: {genetic.calculate_phenotype(population[0])}")