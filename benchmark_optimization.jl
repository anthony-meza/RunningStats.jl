using RunningStats
using Random
using BenchmarkTools

# Set seed for reproducibility
Random.seed!(42)

# Test different sizes to see scaling behavior
test_sizes = [
    (1000, 10),    # Small: 1K samples, 10 features
    (5000, 50),    # Medium: 5K samples, 50 features  
    (10000, 100),  # Large: 10K samples, 100 features
    (20000, 200),  # Very large: 20K samples, 200 features
]

println("Benchmarking update_batch! performance improvements")
println("=" ^ 60)

for (n_samples, n_features) in test_sizes
    println("\nTesting with $n_samples samples, $n_features features:")
    
    # Generate test data
    X = randn(n_samples, n_features)
    
    # Benchmark the optimized version
    estimator = WelfordEstimate()
    
    # Warm up
    update_batch!(estimator, X[1:min(100, n_samples), :])
    
    # Reset for actual benchmark
    estimator = WelfordEstimate()
    
    # Benchmark
    result = @benchmark update_batch!($estimator, $X) setup=(estimator = WelfordEstimate()) evals=1
    
    time_ms = median(result.times) / 1e6  # Convert to milliseconds
    memory_mb = result.memory / 1024^2    # Convert to MB
    
    println("  Time: $(round(time_ms, digits=2)) ms")
    println("  Memory: $(round(memory_mb, digits=2)) MB")
    println("  Time per sample: $(round(time_ms/n_samples * 1000, digits=3)) μs")
end

println("\n" * "=" ^ 60)
println("Optimization complete! The vectorized implementation should show")
println("significant speedups, especially for data with many features.")