class DataInit:
    def __init__(
            self, 
            problem_resolution: float,    
            a_interval: float, 
            b_interval: float,
            initial_population: int,
            max_population: int,
            crossover_threshold: float,
            individual_mutation_threshold: float,
            gene_mutation_threshold: float,
            iterations: int,
            func = (float)
            ):
        self.problem_resolution = problem_resolution    
        self.a_interval= a_interval 
        self.b_interval= b_interval
        self.initial_population= initial_population
        self.max_population= max_population
        self.crossover_threshold= crossover_threshold
        self.individual_mutation_threshold= individual_mutation_threshold
        self.gene_mutation_threshold= gene_mutation_threshold
        self.iterations= iterations
        self.func = func # Función para graficar en matplot
        
    