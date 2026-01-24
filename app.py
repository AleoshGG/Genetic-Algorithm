import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter
from GeneticAlgorithm import GeneticAlgorithm
from models.data_init import DataInit

def func(x):
    argumento_log = np.abs(0.1 + np.cos(7*x)) + 1e-9
    return x * np.cos(x) + np.log(argumento_log) + 2 * np.sin(x/4) + 7 * np.cos(x/8) 

# --- 1. CONFIGURACIÓN E INICIALIZACIÓN ---
data_init = DataInit(
    problem_resolution=0.5,
    a_interval=-1000,
    b_interval=800,
    initial_population=70,
    max_population=100,
    crossover_threshold=0.25,
    individual_mutation_threshold=0.3,
    gene_mutation_threshold=0.31,
    iterations=50, 
    func=func
)

genetic = GeneticAlgorithm(data_init=data_init)
system_res, bytes_length = genetic.system_resolution_and_bits()

print(f"Bytes: {bytes_length} | Resolución del Sistema: {system_res}")

# --- 2. DATOS PARA LA GRÁFICA DE LA FUNCIÓN ---
x_axis = np.arange(data_init.a_interval, data_init.b_interval + system_res, system_res)
y_axis = func(x_axis)

# --- 3. CICLO DE EVOLUCIÓN (Guardando el historial) ---
history = []
population = genetic.generate_initial_population(bytes_length)

print("--- CALCULANDO EVOLUCIÓN ---")

for i in range(data_init.iterations):
    evaluated = genetic.evaluate_population(population)
    pop_x = [ind[1] for ind in evaluated]
    pop_y = [ind[2] for ind in evaluated]
    history.append((pop_x, pop_y, i))
    
    pairs = genetic.generate_pairs(population)
    offspring = genetic.cross(pairs)
    childs = genetic.mutate(offspring)
    population = genetic.prune(population, childs)

# Guardar la última generación
evaluated = genetic.evaluate_population(population)
history.append(([ind[1] for ind in evaluated], [ind[2] for ind in evaluated], data_init.iterations))

print("--- GENERANDO VIDEO (.mp4) ---")

# --- 4. FUNCIÓN DE ANIMACIÓN DEL PAISAJE ---
fig, ax = plt.subplots(figsize=(10, 6))

def update(frame):
    ax.clear()
    pop_x, pop_y, gen = history[frame]
    
    pop_x_arr = np.array(pop_x)
    pop_y_arr = np.array(pop_y)
    
    ax.plot(x_axis, y_axis, color='gray', alpha=0.5, label='Función Objetivo', zorder=1)
    ax.scatter(pop_x_arr, pop_y_arr, color='blue', s=30, alpha=0.4, label='Población', zorder=2)
    
    best_idx = np.argmax(pop_y_arr)
    worst_idx = np.argmin(pop_y_arr)
    
    ax.scatter(pop_x_arr[worst_idx], pop_y_arr[worst_idx], color='red', s=150, marker='X', edgecolors='black', linewidths=1.5, zorder=10, label=f'Peor: {pop_y_arr[worst_idx]:.2f}')
    ax.scatter(pop_x_arr[best_idx], pop_y_arr[best_idx], color='gold', s=200, marker='*', edgecolors='black', linewidths=1.5, zorder=10, label=f'Mejor: {pop_y_arr[best_idx]:.2f}')
    
    ax.set_title(f"Generación {gen} | Mejor Fitness: {pop_y_arr[best_idx]:.4f}")
    ax.set_xlabel("Fenotipo (x)")
    ax.set_ylabel("Aptitud f(x)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right')
    ax.set_xlim(data_init.a_interval, data_init.b_interval)
    ax.set_ylim(min(y_axis) - 50, max(y_axis) + 50)

ani = FuncAnimation(fig, update, frames=len(history), repeat=False)
writer = FFMpegWriter(fps=2)
ani.save("evolucion_lenta.mp4", writer=writer)

print("--- GENERANDO GRÁFICA DE CONVERGENCIA ---")

# --- 5. GRÁFICA DE LÍNEAS DE APTITUD (NUEVO) ---
# Extraemos los datos del historial
generaciones = []
mejores = []
peores = []
promedios = []

for pop_x, pop_y, gen in history:
    mejor = np.max(pop_y)
    peor = np.min(pop_y)
    promedio_dos = (mejor + peor) / 2 # Promedio entre el mejor y el peor
    
    generaciones.append(gen)
    mejores.append(mejor)
    peores.append(peor)
    promedios.append(promedio_dos)

# Crear la gráfica estática
plt.figure(figsize=(10, 6))
plt.plot(generaciones, mejores, color='gold', linewidth=2, label='Mejor Aptitud', marker='o', markersize=4)
plt.plot(generaciones, peores, color='red', linewidth=2, label='Peor Aptitud', marker='x', markersize=4)
plt.plot(generaciones, promedios, color='blue', linestyle='--', linewidth=2, label='Promedio (Mejor y Peor)')

plt.title('Evolución de la Aptitud por Generación')
plt.xlabel('Generación')
plt.ylabel('Aptitud f(x)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='best')
plt.tight_layout()

# Guardar la gráfica como imagen PNG
plt.savefig('convergencia_ga.png')
print("La gráfica de convergencia se guardó como 'convergencia_ga.png'")


print("\n--- FIN DEL ALGORITMO ---")

# --- 6. IMPRIMIR EL MEJOR INDIVIDUO FINAL ---
evaluated.sort(key=lambda x: x[2], reverse=True)
best = evaluated[0][0]
best_x = evaluated[0][1]
best_a = evaluated[0][2]

print(f"Mejor individuo final: {best}")
print(f"\nFenotipo: {best_x}")
print(f"\nAptitud: {best_a}")