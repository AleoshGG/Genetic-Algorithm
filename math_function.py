import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


# 1. DEFINIR EL INTERVALO (límite inferior, límite superior, puntos)
x = np.linspace(20, 22, 50)

# 2. DEFINIR LA FUNCIÓN (Eje Y)
y = x * np.cos(7*x) + np.sin(3*x)

# 3. CREAR LA GRÁFICA
plt.figure(figsize=(10, 6)) # Tamaño de la figura (ancho, alto)
plt.plot(x, y, label=r'$f(x) = x \cos(7x) + \sin(3x)$', color='blue') # 'label' soporta LaTeX

# 4. PERSONALIZACIÓN Y LÍMITES (Intervalos visuales)
plt.title(r'Gráfica de $f(x) = x \cos(7x) + \sin(3x)$ en [20, 22]')
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.grid(True, alpha=0.3)
plt.legend()

# Opcional: Forzar el "zoom" a un intervalo específico visualmente
# plt.xlim(-5, 5) 
# plt.ylim(-20, 20)

plt.show()