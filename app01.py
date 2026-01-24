"""from GeneticAlgorithm import GeneticAlgorithm
from models.data_init import DataInit
import numpy as np


def func(x):
    return x * np.cos(x) + np.log((0.1 + np.cos(7*x))) + 2 * np.sin(x/4) + 7 * np.cos(x/8) 

data_init = DataInit(
    problem_resolution=0.5,
    a_interval=-1000,
    b_interval=800,
    initial_population=10,
    max_population=20,
    crossover_threshold=0.25,
    individual_mutation_threshold=0.2,
    gene_mutation_threshold=0.25,
    iterations=400,
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
print(f"\nFenotipo: {genetic.calculate_phenotype(population[0])}")"""

import matplotlib.pyplot as plt
import numpy as np
from GeneticAlgorithm import GeneticAlgorithm
from models.data_init import DataInit

def func(x):
    return x * np.cos(x) + np.log((0.1 + np.cos(7*x))) + 2 * np.sin(x/4) + 7 * np.cos(x/8) 

def visualizar_generacion(genetic_instance, population, generation, pause_time=0.1):
    """
    Grafica la función real, la población actual (puntos azules)
    y resalta al mejor individuo (estrella roja).
    """
    # Limpiamos la figura anterior para la animación
    plt.clf()
    
    # A) Datos para dibujar la CURVA de la función (el paisaje)
    # Generamos 1000 puntos entre el intervalo A y B para que se vea suave
    a = genetic_instance.data_init.a_interval
    b = genetic_instance.data_init.b_interval
    x_axis = np.linspace(a, b, 1000)
    y_axis = func(x_axis)
    
    plt.plot(x_axis, y_axis, color='gray', alpha=0.5, label='Función Objetivo')

    # B) Datos de la POBLACIÓN (Los exploradores)
    pop_x = []
    pop_y = []
    
    # Convertimos cada individuo (bits) a fenotipo (x) y calculamos su fitness (y)
    for ind in population:
        pheno = genetic_instance.calculate_phenotype(ind)
        fitness = func(pheno)
        pop_x.append(pheno)
        pop_y.append(fitness)
    
    # Dibujamos a toda la población como puntos azules
    plt.scatter(pop_x, pop_y, color='blue', s=30, alpha=0.6, label='Individuos')

    # C) Resaltar al MEJOR CASO (El líder)
    # Buscamos el índice del valor más alto en pop_y
    best_idx = np.argmax(pop_y)
    best_x = pop_x[best_idx]
    best_y = pop_y[best_idx]
    
    plt.scatter(best_x, best_y, color='red', s=150, marker='*', edgecolors='black', label='Mejor Actual')
    
    # Decoración de la gráfica
    plt.title(f"Generación {generation} | Mejor Fitness: {best_y:.4f}")
    plt.xlabel("Fenotipo (x)")
    plt.ylabel("Aptitud f(x)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # Pausa para generar efecto de animación
    plt.draw()
    plt.pause(pause_time)

# --- 3. CONFIGURACIÓN E INICIALIZACIÓN ---

# Nota: Aumenté un poco la población inicial para que se vea mejor en la gráfica
data_init = DataInit(
    problem_resolution=0.5,
    a_interval=-1000,
    b_interval=800,
    initial_population=70,
    max_population=100,
    crossover_threshold=0.25,
    individual_mutation_threshold=0.3,
    gene_mutation_threshold=0.31,
    iterations=200, 
    func=func
)

genetic = GeneticAlgorithm(data_init=data_init)
system_res, bytes_length = genetic.system_resolution_and_bits()

print(f"Bytes: {bytes_length} | Resolución: {system_res}")

# Generar población inicial
population = genetic.generate_initial_population(bytes_length)

print(f"--- INICIO DE EVOLUCIÓN ---")

# Preparamos la ventana de gráfico
plt.figure(figsize=(10, 6))

# --- 4. CICLO PRINCIPAL ---
for i in range(data_init.iterations):
    
    # Lógica del Algoritmo Genético
    pairs = genetic.generate_pairs(population)
    offspring = genetic.cross(pairs)
    childs = genetic.mutate(offspring)
    population = genetic.prune(population, childs)
    
    # --- VISUALIZACIÓN ---
    # Graficamos cada 5 generaciones para que no vaya muy lento
    # (O cambia el % 5 a % 1 si quieres ver todas)
    if i % 1 == 0 or i == 0:
        visualizar_generacion(genetic, population, i)
        
        # Reporte en consola
        best_tuple = genetic.evaluate_population([population[0]])[0]
        print(f"Gen {i}: Mejor = {best_tuple[2]:.5f} (x={best_tuple[1]:.2f})")

# Mantener la gráfica final abierta
print("\n--- FIN DEL ALGORITMO ---")
best = population[0]
best_x = genetic.calculate_phenotype(best)
best_a = func(best_x)

print(f"Mejor individuo final: {best}")
print(f"\nFenotipo: {best_x}")
print(f"\nAptitud: {best_a}")
plt.show()