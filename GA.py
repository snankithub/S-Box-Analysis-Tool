import numpy as np
import math

# Provided functions for nonlinearity computation
def log2n(l):
    """Compute log2 of an integer (only valid for powers of 2)."""
    assert (l & (l - 1)) == 0, "log2n(l) valid only for l=2^n"
    return int(math.log2(l))

def binaryInnerProduct(a, b):
    """Compute binary inner product of two integers."""
    ip = 0
    ab = a & b
    while ab > 0:
        ip ^= (ab & 1)  # XOR the least significant bit
        ab >>= 1
    return ip

def walshTransform(t):
    """Compute the Walsh-Hadamard Transform for a binary sequence."""
    n = log2n(len(t))
    wt = np.zeros(len(t), dtype=int)
    for w in range(len(t)):
        for x in range(len(t)):
            wt[w] += (-1) ** (t[x] ^ binaryInnerProduct(w, x))
    return wt

def nonLinearity(t):
    """Compute non-linearity of a binary sequence."""
    wt = walshTransform(t)
    nl = len(t) // 2 - 0.5 * max(abs(i) for i in wt)
    return nl

def sboxNonlinearity(sbox):
    """Compute minimum and maximum non-linearity of an S-box."""
    n = log2n(len(sbox))
    nlv = np.zeros(2**n - 1, dtype=int)
    for c in range(len(nlv)):
        t = [binaryInnerProduct(c + 1, sbox[i]) for i in range(len(sbox))]
        nlv[c] = nonLinearity(t)
    return min(abs(nlv)), max(abs(nlv))

# Genetic Algorithm Functions
def pmx(parent1, parent2):
    """Partially Mapped Crossover (PMX) for two parents."""
    size = len(parent1)
    cx1, cx2 = sorted(np.random.choice(size, 2, replace=False))
    offspring1 = np.copy(parent1)
    offspring2 = np.copy(parent2)
    
    # Mapping between crossover points
    mapping1 = {parent1[i]: parent2[i] for i in range(cx1, cx2)}
    mapping2 = {parent2[i]: parent1[i] for i in range(cx1, cx2)}
    
    # Resolve conflicts outside crossover segment
    for i in range(size):
        if not (cx1 <= i < cx2):
            while offspring1[i] in mapping1:
                offspring1[i] = mapping1[offspring1[i]]
            while offspring2[i] in mapping2:
                offspring2[i] = mapping2[offspring2[i]]
    return offspring1, offspring2

def swap_mutation(sbox, a_worst):
    """Swap mutation guided by Walsh-Hadamard Transform to improve nonlinearity."""
    sbox = np.copy(sbox)
    size = len(sbox)
    
    # Compute the component function f_a(x) = a · S(x) for the worst a
    fa = np.array([binaryInnerProduct(a_worst, sbox[x]) for x in range(size)])
    wt = walshTransform(fa)
    max_wt_idx = np.argmax(np.abs(wt))
    max_wt = wt[max_wt_idx]
    
    # Find positions contributing to the large W_f(a, u)
    contrib_pos = [x for x in range(size) if (-1) ** (fa[x] ^ binaryInnerProduct(max_wt_idx, x)) == np.sign(max_wt)]
    if len(contrib_pos) < 2:
        # Fallback to random swap if insufficient positions
        i, j = np.random.choice(size, 2, replace=False)
    else:
        # Select two positions to swap that might reduce max |W_f(a, u)|
        i, j = np.random.choice(contrib_pos, 2, replace=False)
    
    sbox[i], sbox[j] = sbox[j], sbox[i]
    return sbox

def generate_initial_population(initial_sbox, pop_size=500):
    """Generate initial population of unique permutations."""
    population = []
    seen = set([tuple(initial_sbox)])
    while len(population) < pop_size:
        perm = np.random.permutation(256)
        perm_tuple = tuple(perm)
        if perm_tuple not in seen:
            seen.add(perm_tuple)
            population.append(perm)
    return population

# Main Genetic Algorithm
def genetic_algorithm(initial_sbox):
    """Optimize S-box for high nonlinearity using a genetic algorithm."""
    population = generate_initial_population(initial_sbox)
    generation = 0
    best_min_nl = 0
    best_sbox = None
    
    while best_min_nl < 108:
        generation += 1
        print(f"\nGeneration {generation}")
        
        # Step 2: Evaluate nonlinearity of each S-box
        nonlinearities = [sboxNonlinearity(sbox) for sbox in population]
        min_nls = [nl[0] for nl in nonlinearities]
        
        # Find the best S-box in this generation
        best_idx = np.argmax(min_nls)
        current_best_min_nl = min_nls[best_idx]
        if current_best_min_nl > best_min_nl:
            best_min_nl = current_best_min_nl
            best_sbox = population[best_idx]
        print(f"Best minimum nonlinearity: {best_min_nl}")
        
        if best_min_nl >= 108:
            break
        
        # Step 3: Elitist Selection
        threshold = 102
        parents = [population[i] for i in range(len(population)) if min_nls[i] > threshold]
        if not parents:
            # If no S-box exceeds threshold, select the top one
            top_idx = np.argmax(min_nls)
            parents = [population[top_idx]]
        print(f"Number of parents selected: {len(parents)}")
        
        # Step 4: Generate new population
        new_population = [best_sbox]  # Elitism: carry over the best
        seen = set([tuple(best_sbox)])
        
        # Find the component 'a' with the lowest nonlinearity for mutation guidance
        n = log2n(len(best_sbox))
        nlv = np.zeros(2**n - 1, dtype=int)
        for c in range(len(nlv)):
            t = [binaryInnerProduct(c + 1, best_sbox[i]) for i in range(len(best_sbox))]
            nlv[c] = nonLinearity(t)
        a_worst = np.argmin(np.abs(nlv)) + 1  # +1 because c starts from 0
        
        while len(new_population) < 500:
            parent1, parent2 = np.random.choice(parents, 2, replace=True) if len(parents) > 1 else (parents[0], parents[0])
            offspring1, offspring2 = pmx(parent1, parent2)
            
            # Apply mutation to both offspring
            offspring1 = swap_mutation(offspring1, a_worst)
            offspring2 = swap_mutation(offspring2, a_worst)
            
            for offspring in [offspring1, offspring2]:
                offspring_tuple = tuple(offspring)
                if offspring_tuple not in seen and len(new_population) < 500:
                    seen.add(offspring_tuple)
                    new_population.append(offspring)
        
        population = new_population
    
    return best_sbox, best_min_nl

# Execute the genetic algorithm
if __name__ == "__main__":
    print("Starting genetic algorithm... be patient, this will take a while...")
    
    # Step 1: Define the initial S-box
    initial_sbox = np.array([
        76, 90, 254, 156, 196, 146, 188, 201, 9, 161, 32, 75, 21, 49, 197, 47,
        135, 238, 2, 187, 233, 122, 180, 162, 22, 190, 120, 38, 50, 130, 114, 125,
        174, 94, 40, 48, 230, 36, 177, 118, 62, 232, 214, 158, 16, 8, 53, 60,
        144, 165, 99, 74, 185, 35, 15, 29, 101, 82, 20, 154, 171, 100, 251, 176,
        133, 245, 105, 182, 97, 224, 152, 191, 91, 30, 26, 1, 59, 209, 64, 184,
        183, 253, 178, 12, 208, 44, 204, 145, 167, 210, 28, 160, 45, 19, 83, 131,
        173, 92, 189, 234, 86, 27, 124, 159, 149, 223, 84, 212, 241, 179, 142, 69,
        206, 169, 54, 106, 213, 221, 244, 202, 207, 227, 163, 13, 192, 186, 6, 252,
        110, 220, 93, 68, 88, 143, 243, 215, 108, 164, 138, 170, 78, 103, 113, 23,
        246, 231, 70, 31, 25, 172, 193, 140, 104, 222, 216, 79, 229, 157, 111, 166,
        7, 199, 137, 3, 148, 175, 151, 17, 5, 42, 116, 134, 37, 147, 112, 247,
        80, 55, 205, 39, 250, 237, 129, 218, 87, 203, 51, 14, 132, 24, 18, 119,
        115, 225, 57, 248, 61, 168, 71, 181, 226, 211, 195, 85, 121, 239, 153, 73,
        109, 219, 66, 228, 255, 43, 34, 95, 150, 0, 4, 58, 63, 126, 107, 141,
        10, 67, 72, 33, 217, 11, 240, 249, 128, 136, 235, 96, 123, 117, 127, 41,
        98, 242, 65, 77, 52, 139, 198, 194, 236, 81, 155, 102, 46, 89, 56, 200
    ])
    
    # Verify initial S-box
    min_nl, max_nl = sboxNonlinearity(initial_sbox)
    print(f"Initial S-box: min_nl = {min_nl}, max_nl = {max_nl}")
    
    # Run the genetic algorithm
    optimized_sbox, optimized_min_nl = genetic_algorithm(initial_sbox)
    
    # Output the result
    final_min_nl, final_max_nl = sboxNonlinearity(optimized_sbox)
    print(f"\nOptimized S-box achieved:")
    print(f"Minimum nonlinearity: {final_min_nl}")
    print(f"Maximum nonlinearity: {final_max_nl}")
    print(f"Optimized S-box: {optimized_sbox.tolist()}")
