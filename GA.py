import numpy as np
import math

# Fast Walsh-Hadamard Transform (FWHT)
def fwht(arr):
    """Compute the Fast Walsh-Hadamard Transform (FWHT)."""
    n = len(arr)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(h):
                x = arr[i + j]
                y = arr[i + j + h]
                arr[i + j] = x + y
                arr[i + j + h] = x - y
        h *= 2
    return arr

# Compute binary inner product
def binaryInnerProduct(a, b):
    """Binary inner product of two integers."""
    return bin(a & b).count('1') % 2

# Compute nonlinearity using FWHT
def nonLinearity(t):
    """Compute the non-linearity using FWHT."""
    n = len(t)
    wt = np.array([(-1) ** t[x] for x in range(n)], dtype=np.int32)
    fwht(wt)
    return int(n / 2 - 0.5 * max(abs(wt)))

# Compute S-box non-linearity
def sboxNonlinearity(sbox):
    """Compute minimum and maximum non-linearity of an S-box."""
    n = int(math.log2(len(sbox)))
    num_components = 2 ** n - 1
    nonlinearities = np.zeros(num_components, dtype=np.int32)
    
    for c in range(num_components):
        t = np.array([binaryInnerProduct(c + 1, sbox[i]) for i in range(len(sbox))])
        nonlinearities[c] = nonLinearity(t)
    
    return min(nonlinearities), max(nonlinearities)

# Partially Mapped Crossover (PMX)
def pmx(parent1, parent2):
    """Partially Mapped Crossover (PMX)"""
    size = len(parent1)
    cx1, cx2 = sorted(np.random.choice(size, 2, replace=False))
    
    offspring1, offspring2 = np.copy(parent1), np.copy(parent2)
    mapping1 = {parent1[i]: parent2[i] for i in range(cx1, cx2)}
    mapping2 = {parent2[i]: parent1[i] for i in range(cx1, cx2)}

    for i in range(size):
        if i < cx1 or i >= cx2:
            while offspring1[i] in mapping1:
                offspring1[i] = mapping1[offspring1[i]]
            while offspring2[i] in mapping2:
                offspring2[i] = mapping2[offspring2[i]]
    
    return offspring1, offspring2

# Swap Mutation with Guided Optimization
def swap_mutation(sbox):
    """Swap mutation to improve non-linearity."""
    size = len(sbox)
    i, j = np.random.choice(size, 2, replace=False)
    sbox[i], sbox[j] = sbox[j], sbox[i]
    return sbox

# Generate initial population with unique permutations
def generate_initial_population(size=10000):
    """Generate an initial population of unique S-boxes."""
    population = set()
    while len(population) < size:
        perm = tuple(np.random.permutation(256))
        population.add(perm)
    return [np.array(sbox) for sbox in population]

# HawkBoost Algorithm (Placeholder)
def hawkboost_algorithm(best_sbox, best_min_nl):
    """HawkBoost Algorithm to enhance the solution."""
    # Implement the HawkBoost Algorithm based on the resources provided
    # This is a placeholder function and should be replaced with the actual implementation
    return best_sbox, best_min_nl

# Genetic Algorithm for S-Box Optimization
def genetic_algorithm(max_generations=25):
    """Optimize an S-box for high nonlinearity using Genetic Algorithm."""
    population = generate_initial_population()
    best_min_nl = 0
    best_sbox = None

    for generation in range(max_generations):
        nonlinearities = [sboxNonlinearity(sbox) for sbox in population]
        min_nls = [nl[0] for nl in nonlinearities]

        # Select best individual
        best_idx = np.argmax(min_nls)
        current_best_min_nl = min_nls[best_idx]

        if current_best_min_nl > best_min_nl:
            best_min_nl = current_best_min_nl
            best_sbox = population[best_idx]

        print(f"Generation {generation + 1}: Best min non-linearity = {best_min_nl}")

        # Elitist selection
        elite_threshold = 102
        parents_indices = [i for i in range(len(population)) if min_nls[i] > elite_threshold]

        if not parents_indices:
            parents_indices = [best_idx]

        # Crossover & Mutation
        new_population = [best_sbox]
        seen = {tuple(best_sbox)}

        while len(new_population) < 1000:
            parent1_idx, parent2_idx = np.random.choice(parents_indices, 2, replace=True)
            parent1, parent2 = population[parent1_idx], population[parent2_idx]
            offspring1, offspring2 = pmx(parent1, parent2)

            # Apply mutation
            offspring1, offspring2 = swap_mutation(offspring1), swap_mutation(offspring2)

            for offspring in [offspring1, offspring2]:
                if tuple(offspring) not in seen and len(new_population) < 1000:
                    seen.add(tuple(offspring))
                    new_population.append(offspring)

        population = new_population

        # Integrate HawkBoost Algorithm
        if best_min_nl >= 104:
            best_sbox, best_min_nl = hawkboost_algorithm(best_sbox, best_min_nl)
            if best_min_nl >= 110:
                break

    return best_sbox, best_min_nl

# Run Genetic Algorithm
if __name__ == "__main__":
    print("Starting optimized genetic algorithm...")

    # Generate initial random S-box
    initial_sbox = np.random.permutation(256)
    
    # Check initial non-linearity
    min_nl, max_nl = sboxNonlinearity(initial_sbox)
    print(f"Initial S-box: min_nl = {min_nl}, max_nl = {max_nl}")

    # Run optimization
    optimized_sbox, optimized_min_nl = genetic_algorithm()

    # Output results
    final_min_nl, final_max_nl = sboxNonlinearity(optimized_sbox)
    print("\nOptimized S-box achieved:")
    print(f"Minimum non-linearity: {final_min_nl}")
    print(f"Maximum non-linearity: {final_max_nl}")
    print(f"Optimized S-box: {optimized_sbox.tolist()}")
