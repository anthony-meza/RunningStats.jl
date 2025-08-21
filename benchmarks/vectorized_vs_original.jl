using RunningStats
using Random
using BenchmarkTools

# Set seed for reproducibility
Random.seed!(42)

# Original implementation for comparison (single-sample loop)
function update_batch_original!(wc::WelfordEstimate{T}, X::AbstractMatrix{<:Real}) where T
    X_converted = convert(Matrix{T}, X)
    n_samples, n_features = size(X_converted)
    
    if wc.n == 0
        RunningStats.initialize!(wc, n_features)
    elseif n_features != wc.n_features
        throw(DimensionMismatch("Expected $(wc.n_features) features, got $n_features"))
    end
    
    # Original loop-based implementation
    for i in 1:n_samples
        sample = view(X_converted, i, :)
        RunningStats.update_single!(wc, sample)
    end
    
    return RunningStats.get_statistics(wc)
end

# Test different sizes to see scaling behavior
test_sizes = [
    (1000, 10),    # Small: 1K samples, 10 features
    (5000, 50),    # Medium: 5K samples, 50 features  
    (10000, 100),  # Large: 10K samples, 100 features
    (20000, 200),  # Very large: 20K samples, 200 features
]

println("Benchmarking: Vectorized vs Original Implementation")
println("=" ^ 70)

for (n_samples, n_features) in test_sizes
    println("\nTesting with $n_samples samples, $n_features features:")
    
    # Generate test data
    X = randn(n_samples, n_features)
    
    # Benchmark original implementation
    println("  Original (loop-based):")
    result_orig = @benchmark update_batch_original!(estimator, $X) setup=(estimator = WelfordEstimate()) evals=1
    time_orig_ms = median(result_orig.times) / 1e6
    memory_orig_mb = result_orig.memory / 1024^2
    println("    Time: $(round(time_orig_ms, digits=2)) ms")
    println("    Memory: $(round(memory_orig_mb, digits=2)) MB")
    
    # Benchmark vectorized implementation  
    println("  Vectorized (new):")
    result_new = @benchmark update_batch!(estimator, $X) setup=(estimator = WelfordEstimate()) evals=1
    time_new_ms = median(result_new.times) / 1e6
    memory_new_mb = result_new.memory / 1024^2
    println("    Time: $(round(time_new_ms, digits=2)) ms")
    println("    Memory: $(round(memory_new_mb, digits=2)) MB")
    
    # Calculate speedup
    speedup = time_orig_ms / time_new_ms
    memory_ratio = memory_orig_mb / memory_new_mb
    
    println("  Performance gain:")
    println("    Speedup: $(round(speedup, digits=2))x faster")
    println("    Memory ratio: $(round(memory_ratio, digits=2))x ($(memory_ratio > 1 ? "less" : "more") memory)")
    
    # Verify results are identical
    est1 = WelfordEstimate()
    est2 = WelfordEstimate() 
    
    stats1 = update_batch_original!(est1, X)
    stats2 = update_batch!(est2, X)
    
    max_diff = maximum(abs.(stats1.covariance - stats2.covariance))
    println("    Max covariance difference: $(max_diff) (should be ~0)")
end

println("\n" * "=" ^ 70)
println("Summary: The vectorized implementation should show significant")
println("speedups, especially for data with many features, by eliminating")
println("the loop and using efficient matrix operations.")