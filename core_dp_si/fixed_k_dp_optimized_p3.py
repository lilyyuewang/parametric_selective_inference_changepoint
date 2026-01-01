import numpy as np
import time


def ssq_vectorized(j_array, i, sum_x, sum_x_sq):
    """
    PRIORITY 3: Vectorized computation of SSE for multiple segments at once.
    
    Computes SSE(j, i) for all j in j_array simultaneously.
    
    Args:
        j_array: Array of starting positions [j1, j2, ..., jn]
        i: Common ending position
        sum_x: Cumulative sum array
        sum_x_sq: Cumulative sum of squares array
    
    Returns:
        sse_array: Array of SSE values for each j
    
    Time: O(len(j_array)) = O(n) instead of O(n) individual calls
    Benefit: Vectorized operations are much faster in NumPy
    """
    j_array = np.asarray(j_array)
    n_candidates = len(j_array)
    
    # Vectorized computation for j > 0 cases
    mask_positive = j_array > 0
    sse_array = np.zeros(n_candidates)
    
    # Handle j > 0 cases (vectorized)
    if np.any(mask_positive):
        j_pos = j_array[mask_positive]
        
        # Vectorized mean computation
        sum_diff = sum_x[i] - sum_x[j_pos - 1]
        length = i - j_pos + 1
        mu_ji = sum_diff / length
        
        # Vectorized SSE computation
        sse = (sum_x_sq[i] - sum_x_sq[j_pos - 1] 
               - length * mu_ji ** 2)
        sse_array[mask_positive] = np.maximum(0, sse)
    
    # Handle j = 0 cases (vectorized)
    mask_zero = j_array == 0
    if np.any(mask_zero):
        sse = sum_x_sq[i] - sum_x[i] ** 2 / (i + 1)
        sse_array[mask_zero] = max(0, sse)
    
    return sse_array


def ssq_matrix_lazy(j, i, sum_x_matrix, n):
    """
    PRIORITY 1: Compute SSE matrix on-the-fly without storing all n×n matrices.
    (Same as before - not vectorized because it's only called for selected j's)
    """
    if j > 0:
        indicator_i = sum_x_matrix[i]
        indicator_j_minus_1 = sum_x_matrix[j - 1]
        indicator_diff = indicator_i - indicator_j_minus_1
        
        segment_length = i - j + 1
        muji_matrix = indicator_diff / segment_length
        
        outer_product_i = np.dot(indicator_i, indicator_i.T)
        outer_product_j = np.dot(indicator_j_minus_1, indicator_j_minus_1.T)
        
        dji_matrix = (outer_product_i - outer_product_j 
                      - segment_length * np.dot(muji_matrix, muji_matrix.T))
    else:
        indicator_i = sum_x_matrix[i]
        segment_length = i + 1
        outer_product_i = np.dot(indicator_i, indicator_i.T)
        dji_matrix = outer_product_i - (outer_product_i / segment_length)
    
    return dji_matrix


def fill_dp_matrix_optimized_p3(data, S, J, K, n, constraint_mode='competitive', 
                                  tolerance=1e-10, relative_threshold=0.01):
    """
    PRIORITY 1 + PRIORITY 2 + PRIORITY 3:
    Lazy matrix computation + selective constraints + vectorization
    
    NEW in Priority 3:
    - Vectorized SSE computation for all candidate j's at once
    - Faster min/argmin operations
    - Reduced Python loop overhead
    
    Expected speedup: 5-10× for the DP filling phase
    
    Returns:
        list_condition_matrix: Constraint matrices
        constraint_stats: Statistics about constraint reduction
        timing_stats: Detailed timing breakdown (new)
    """
    list_matrix = []
    list_condition_matrix = []
    
    # Statistics tracking
    constraint_stats = {
        'total_candidates': 0,
        'stored_constraints': 0,
        'reduction_by_level': [],
    }
    
    # Timing statistics (Priority 3)
    timing_stats = {
        'precomputation': 0,
        'initialization': 0,
        'dp_vectorized': 0,
        'constraint_generation': 0,
        'total': 0
    }
    
    start_total = time.time()

    # Scalar cumulative sums: O(n) space
    sum_x = np.zeros(n, dtype=np.float64)
    sum_x_sq = np.zeros(n, dtype=np.float64)
    sum_x_matrix = []

    shift = 0

    # ============================================
    # PHASE 1: Initialization (k=0)
    # ============================================
    start_init = time.time()
    
    for i in range(n):
        if i == 0:
            sum_x[0] = data[0] - shift
            sum_x_sq[0] = (data[0] - shift) ** 2

            e_n_0 = np.zeros(n)
            e_n_0[0] = 1
            e_n_0 = e_n_0.reshape((n, 1))

            sum_x_matrix.append(e_n_0)

        else:
            sum_x[i] = sum_x[i - 1] + data[i] - shift
            sum_x_sq[i] = sum_x_sq[i - 1] + (data[i] - shift) ** 2

            e_n_i = np.zeros(n)
            e_n_i[i] = 1
            e_n_i = e_n_i.reshape((n, 1))

            sum_x_matrix.append(sum_x_matrix[i - 1] + e_n_i)

        # Compute initial SSE (can use vectorized version but not much gain here)
        if i == 0:
            S[0][i] = max(0, sum_x_sq[i] - sum_x[i] ** 2 / (i + 1))
        else:
            S[0][i] = max(0, sum_x_sq[i] - sum_x[i] ** 2 / (i + 1))
        
        J[0][i] = 0
        list_matrix.append(ssq_matrix_lazy(0, i, sum_x_matrix, n))
    
    timing_stats['initialization'] = time.time() - start_init

    # ============================================
    # PHASE 2: VECTORIZED FORWARD DP (k=1 to K-1)
    # ============================================
    start_dp = time.time()
    
    for k in range(1, K):
        if k < K - 1:
            imin = max(1, k)
        else:
            imin = n - 1

        imax = n - 1
        new_list_matrix = [None] * n
        
        level_candidates = 0
        level_stored = 0

        for i in range(imin, imax + 1):
            # Initialize with no changepoint
            S[k][i] = S[k - 1][i - 1]
            J[k][i] = i

            # ============================================
            # PRIORITY 3: VECTORIZED SSE COMPUTATION
            # ============================================
            
            # Generate all candidate j values
            jmin = k
            j_candidates = np.arange(i, jmin - 1, -1)  # [i, i-1, ..., jmin]
            
            # Vectorized SSE computation for ALL candidates at once
            sse_values = ssq_vectorized(j_candidates, i, sum_x, sum_x_sq)
            
            # Get previous DP costs (vectorized array indexing)
            j_minus_1 = j_candidates - 1
            j_minus_1[j_minus_1 < 0] = 0  # Handle j=0 case
            prev_costs = np.zeros(len(j_candidates))
            prev_costs[1:] = S[k - 1][j_minus_1[1:]]  # j > 0 cases
            prev_costs[0] = S[k - 1][i - 1]  # j = i case (no changepoint)
            
            # Vectorized total cost computation
            total_costs = sse_values + prev_costs
            
            # Find optimal j using vectorized operations
            min_idx = np.argmin(total_costs)
            min_cost = total_costs[min_idx]
            optimal_j = j_candidates[min_idx]
            
            # Update DP tables if better than initialization
            if min_cost < S[k][i]:
                S[k][i] = min_cost
                J[k][i] = optimal_j
            
            # ============================================
            # CONSTRAINT GENERATION (Priority 2)
            # ============================================
            
            # Build alternatives list efficiently
            alternatives = []
            
            # First alternative: no changepoint
            matrix_Y = list_matrix[i - 1]
            new_list_matrix[i] = matrix_Y
            alternatives.append((matrix_Y, S[k - 1][i - 1], i))
            
            # Remaining alternatives (only compute matrices for stored ones)
            for idx, j in enumerate(j_candidates):
                if j == i:  # Skip the "no changepoint" case (already added)
                    continue
                
                cost = total_costs[idx]
                
                # Decide if we'll store this constraint
                should_compute_matrix = False
                
                if constraint_mode == 'all':
                    should_compute_matrix = True
                elif constraint_mode == 'competitive':
                    if S[k][i] < tolerance:
                        should_compute_matrix = (abs(cost - S[k][i]) < tolerance)
                    else:
                        relative_diff = (cost - S[k][i]) / S[k][i]
                        should_compute_matrix = (relative_diff <= relative_threshold)
                elif constraint_mode == 'minimal':
                    should_compute_matrix = (j == J[k][i])
                elif constraint_mode == 'adaptive':
                    cost_range = max(total_costs) - min(total_costs)
                    if cost_range < tolerance:
                        should_compute_matrix = True
                    else:
                        adaptive_threshold = min(
                            relative_threshold * S[k][i],
                            0.1 * cost_range
                        )
                        should_compute_matrix = (cost - S[k][i] <= adaptive_threshold)
                
                # Only compute matrix if we'll use it
                if should_compute_matrix or j == optimal_j:
                    matrix_Y = list_matrix[j - 1] if j > 0 else list_matrix[0]
                    matrix_Z = ssq_matrix_lazy(j, i, sum_x_matrix, n)
                    matrix_Y_plus_Z = matrix_Y + matrix_Z
                    
                    if j == optimal_j:
                        new_list_matrix[i] = matrix_Y_plus_Z
                    
                    alternatives.append((matrix_Y_plus_Z, cost, j))
            
            # Store constraints selectively
            matrix_X = new_list_matrix[i]
            level_candidates += len(alternatives)
            
            for matrix_Y_plus_Z, alternative_cost, j_candidate in alternatives:
                should_store = False
                
                if constraint_mode == 'all':
                    should_store = True
                elif constraint_mode == 'competitive':
                    if S[k][i] < tolerance:
                        should_store = (abs(alternative_cost - S[k][i]) < tolerance)
                    else:
                        relative_diff = (alternative_cost - S[k][i]) / S[k][i]
                        should_store = (relative_diff <= relative_threshold)
                elif constraint_mode == 'minimal':
                    should_store = (j_candidate == J[k][i])
                elif constraint_mode == 'adaptive':
                    costs = [alt[1] for alt in alternatives]
                    cost_range = max(costs) - min(costs)
                    if cost_range < tolerance:
                        should_store = True
                    else:
                        adaptive_threshold = min(
                            relative_threshold * S[k][i],
                            0.1 * cost_range
                        )
                        should_store = (alternative_cost - S[k][i] <= adaptive_threshold)
                
                if should_store:
                    constraint = matrix_X - matrix_Y_plus_Z
                    list_condition_matrix.append(constraint)
                    level_stored += 1
        
        # Update statistics
        constraint_stats['total_candidates'] += level_candidates
        constraint_stats['stored_constraints'] += level_stored
        
        if level_candidates > 0:
            reduction_ratio = 1.0 - (level_stored / level_candidates)
            constraint_stats['reduction_by_level'].append({
                'k': k,
                'candidates': level_candidates,
                'stored': level_stored,
                'reduction': reduction_ratio
            })

        list_matrix = new_list_matrix[:]
    
    timing_stats['dp_vectorized'] = time.time() - start_dp
    timing_stats['total'] = time.time() - start_total

    return list_condition_matrix, constraint_stats, timing_stats


def dp_si_optimized_p3(data, n_segments, constraint_mode='competitive', 
                        tolerance=1e-10, relative_threshold=0.01, 
                        verbose=False, show_timing=False):
    """
    PRIORITY 1 + PRIORITY 2 + PRIORITY 3:
    Optimized DP with lazy computation, selective constraints, and vectorization.
    
    NEW in Priority 3:
    - show_timing: Display detailed timing breakdown
    
    Expected improvements over P1+P2:
    - Speed: 5-10× faster for DP phase
    - Memory: Same as P1+P2
    - Results: Identical
    
    Args:
        data: Input time series of length n
        n_segments: Number of segments K (K-1 changepoints)
        constraint_mode: How to select constraints to store
        tolerance: Absolute tolerance for numerical comparisons
        relative_threshold: For 'competitive' mode
        verbose: Print constraint reduction statistics
        show_timing: Print detailed timing breakdown (Priority 3)
    
    Returns:
        segment_index: Array indicating which segment each point belongs to
        list_condition_matrix: Constraint matrices for selective inference
        sg_results: List of segment boundary positions
        [stats]: Optional dictionary with statistics (if verbose or show_timing)
    """
    n = len(data)

    S = np.zeros((n_segments, n))
    J = np.zeros((n_segments, n))

    start_total = time.time()
    
    list_condition_matrix, constraint_stats, timing_stats = fill_dp_matrix_optimized_p3(
        data, S, J, n_segments, n,
        constraint_mode=constraint_mode,
        tolerance=tolerance,
        relative_threshold=relative_threshold
    )

    # ============================================
    # PHASE 3: Backtracking
    # ============================================
    start_backtrack = time.time()
    
    segment_index = np.zeros(n)
    sg_results = []
    segment_right = n - 1

    for segment in range(n_segments - 1, -1, -1):
        segment_left = int(J[segment][segment_right])
        sg_results.append(segment_right + 1)

        for i in range(segment_left, segment_right + 1):
            segment_index[i] = segment + 1

        if segment > 0:
            segment_right = segment_left - 1

    sg_results.append(0)
    
    timing_stats['backtracking'] = time.time() - start_backtrack
    timing_stats['total_with_backtrack'] = time.time() - start_total
    
    if verbose:
        print_constraint_stats(constraint_stats)
    
    if show_timing:
        print_timing_stats(timing_stats, n, n_segments)
    
    if verbose or show_timing:
        stats = {**constraint_stats, 'timing': timing_stats}
        return segment_index, list_condition_matrix, list(reversed(sg_results)), stats
    else:
        return segment_index, list_condition_matrix, list(reversed(sg_results))


def print_constraint_stats(stats):
    """Print statistics about constraint reduction."""
    total = stats['total_candidates']
    stored = stats['stored_constraints']
    
    if total > 0:
        overall_reduction = (1.0 - stored / total) * 100
        print(f"\n{'='*60}")
        print("CONSTRAINT STORAGE STATISTICS")
        print(f"{'='*60}")
        print(f"Total candidate constraints: {total}")
        print(f"Stored constraints:          {stored}")
        print(f"Reduction:                   {overall_reduction:.1f}%")
        print(f"{'='*60}\n")


def print_timing_stats(timing, n, K):
    """Print detailed timing breakdown (Priority 3)."""
    print(f"\n{'='*60}")
    print("TIMING BREAKDOWN (Priority 3)")
    print(f"{'='*60}")
    print(f"Problem size: n={n}, K={K}")
    print(f"{'-'*60}")
    print(f"Initialization:       {timing['initialization']:>8.4f}s")
    print(f"DP (vectorized):      {timing['dp_vectorized']:>8.4f}s")
    print(f"Backtracking:         {timing['backtracking']:>8.4f}s")
    print(f"{'-'*60}")
    print(f"Total:                {timing['total_with_backtrack']:>8.4f}s")
    print(f"{'='*60}\n")


# ============================================
# Comparison and Benchmarking
# ============================================

def benchmark_all_priorities(data, n_segments, num_trials=3):
    """
    Benchmark all priority levels: Original, P1, P1+P2, P1+P2+P3.
    """
    print(f"\n{'='*70}")
    print(f"BENCHMARK ALL PRIORITIES: n={len(data)}, K={n_segments}")
    print(f"{'='*70}\n")
    
    results = []
    
    # Try to import and test each version
    implementations = []
    
    # Original
    try:
        from core_dp_si.fixed_k_dp import dp_si
        implementations.append(('Original', dp_si, {}))
    except ImportError:
        print("Warning: Original implementation not found")
    
    # P1
    try:
        from core_dp_si.fixed_k_dp_optimized import dp_si_optimized
        implementations.append(('P1 (lazy)', dp_si_optimized, {}))
    except ImportError:
        print("Warning: P1 implementation not found")
    
    # P1+P2
    try:
        from core_dp_si.fixed_k_dp_optimized_p2 import dp_si_optimized_p2
        implementations.append(('P1+P2 (selective)', dp_si_optimized_p2, 
                               {'constraint_mode': 'competitive', 'relative_threshold': 0.01}))
    except ImportError:
        print("Warning: P1+P2 implementation not found")
    
    # P1+P2+P3
    implementations.append(('P1+P2+P3 (vectorized)', dp_si_optimized_p3,
                           {'constraint_mode': 'competitive', 'relative_threshold': 0.01}))
    
    # Run benchmarks
    baseline_cp = None
    baseline_seg = None
    
    for name, func, params in implementations:
        print(f"{name}:")
        print(f"{'-'*70}")
        
        times = []
        for trial in range(num_trials):
            start = time.time()
            try:
                result = func(data, n_segments, **params)
                elapsed = time.time() - start
                times.append(elapsed)
            except Exception as e:
                print(f"  Error: {e}")
                break
        
        if times:
            seg, const, cp = result[:3]
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            if baseline_cp is None:
                baseline_cp = cp
                baseline_seg = seg
            
            cp_match = (cp == baseline_cp)
            seg_match = np.allclose(seg, baseline_seg)
            
            results.append({
                'name': name,
                'time': avg_time,
                'std': std_time,
                'constraints': len(const),
                'correct': cp_match and seg_match
            })
            
            print(f"  Time: {avg_time:.4f} ± {std_time:.4f}s")
            print(f"  Constraints: {len(const)}")
            print(f"  Correct: {'✓' if cp_match and seg_match else '✗'}")
            print()
    
    # Summary table
    if results:
        print(f"{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}\n")
        
        baseline_time = results[0]['time']
        
        print(f"{'Version':<25} {'Time':<12} {'Speedup':<10} {'Constraints':<12} {'Correct'}")
        print(f"{'-'*70}")
        
        for r in results:
            speedup = baseline_time / r['time']
            check = '✓' if r['correct'] else '✗'
            print(f"{r['name']:<25} {r['time']:>8.4f}s  {speedup:>6.2f}×    "
                  f"{r['constraints']:<12} {check}")
        
        print()


if __name__ == "__main__":
    print("="*70)
    print("PRIORITY 3 IMPLEMENTATION: VECTORIZATION")
    print("="*70)
    
    # Test on simple example
    print("\nSimple Example:")
    print("-"*70)
    data = [1, 1, 1, 5, 5, 5]
    K = 2
    
    seg, const, cp, stats = dp_si_optimized_p3(
        data, K,
        constraint_mode='competitive',
        relative_threshold=0.01,
        verbose=True,
        show_timing=True
    )
    
    print(f"Detected changepoints: {cp}")
    print(f"Number of constraints stored: {len(const)}")
    
    # Benchmark on larger data
    print(f"\n{'='*70}")
    print("BENCHMARK: Comparing all priority levels")
    print(f"{'='*70}")
    
    np.random.seed(42)
    test_data = np.concatenate([
        np.random.normal(1, 0.3, 67),
        np.random.normal(3, 0.3, 66),
        np.random.normal(5, 0.3, 67)
    ])
    
    benchmark_all_priorities(test_data, K=3, num_trials=5)

