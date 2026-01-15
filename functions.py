import math

class SystemSolution:
    def __init__(self, a_interval, b_interval):
        self.a_interval = a_interval
        self.b_interval = b_interval
        self.func_range = b_interval - (a_interval)
        self.system_resolution = None

    # Función que retorna la resolución del sistema y el número de bits a usar. Recibe el rango de la función y el valor de la resolución del problema.
    def system_resolution_and_bits(self, problem_resolution: float, func_range: float = None) -> tuple[float, int]:
        
        if func_range is None:
            func_range = self.func_range

        # Encontrar los puntos del problema
        problem_points = math.ceil(func_range / problem_resolution) + 1
        bits = math.ceil(math.log2(problem_points))

        system_points = 2**bits
        system_resolution = func_range / (system_points - 1)    
        
        self.system_resolution = system_resolution

        return system_resolution, bits
    
    # Función para hallar el fenotipo
    def phenotype(self, index = int) -> float:
        if self.system_resolution is None:
            raise ValueError("The solution to the system has not been found")

        return self.a_interval + (index * self.system_resolution)


        




    