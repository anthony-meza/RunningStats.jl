using RunningStats
using Statistics
using Random
using BenchmarkTools

# Set seed for reproducibility
Random.seed!(42)

# Test with a reasonably large dataset to see the performance gap
n_samples, n_features = 10000, 100
X = randn(n_samples, n_features)

println("Performance comparison: RunningStats vs Julia stdlib")
println("Data size: $n_samples samples × $n_features features")
println("=" ^ 60)

# Benchmark Julia's standard cov()
println("Julia stdlib cov():")
result_stdlib = @benchmark cov($X) evals=1
time_stdlib = median(result_stdlib.times) / 1e6
memory_stdlib = result_stdlib.memory / 1024^2
println("  Time: $(round(time_stdlib, digits=2)) ms")
println("  Memory: $(round(memory_stdlib, digits=2)) MB")
println("  Allocations: $(result_stdlib.allocs)")

# Benchmark RunningStats update_batch!
println("\nRunningStats update_batch!:")
result_rs = @benchmark update_batch!(estimator, $X) setup=(estimator = WelfordEstimate()) evals=1
time_rs = median(result_rs.times) / 1e6
memory_rs = result_rs.memory / 1024^2
println("  Time: $(round(time_rs, digits=2)) ms")
println("  Memory: $(round(memory_rs, digits=2)) MB")
println("  Allocations: $(result_rs.allocs)")

# Benchmark just getting covariance (no full stats)
println("\nRunningStats get_covariance only:")
result_cov_only = @benchmark get_covariance(estimator) setup=(estimator = WelfordEstimate(); update_batch!(estimator, $X)) evals=1
time_cov_only = median(result_cov_only.times) / 1e6
memory_cov_only = result_cov_only.memory / 1024^2
println("  Time: $(round(time_cov_only, digits=2)) ms")
println("  Memory: $(round(memory_cov_only, digits=2)) MB")
println("  Allocations: $(result_cov_only.allocs)")

# Performance ratios
println("\nPerformance ratios (RunningStats / stdlib):")
println("  Time ratio: $(round(time_rs / time_stdlib, digits=2))x")
println("  Memory ratio: $(round(memory_rs / memory_stdlib, digits=2))x")

# Verify results are close
estimator = WelfordEstimate()
update_batch!(estimator, X)
cov_rs = get_covariance(estimator)
cov_stdlib = cov(X)

max_diff = maximum(abs.(cov_rs - cov_stdlib))
println("  Max difference: $(max_diff)")

println("\n" * "=" ^ 60)
println("Analysis: Need to identify why RunningStats is slower/more memory-intensive")