using RunningStats
using Statistics
using Random
using BenchmarkTools

# Set seed for reproducibility
Random.seed!(42)

# Test with the same size that showed the performance gap
n_samples, n_features = 10000, 100
X = randn(n_samples, n_features)

println("Isolating the covariance computation bottleneck")
println("Data size: $n_samples samples × $n_features features")
println("=" ^ 60)

# First, set up the estimator with data
estimator = WelfordEstimate()
update_batch!(estimator, X)

println("1. Julia stdlib cov():")
result_stdlib = @benchmark cov($X) evals=1
time_stdlib = median(result_stdlib.times) / 1e6
memory_stdlib = result_stdlib.memory / 1024^2
println("   Time: $(round(time_stdlib, digits=2)) ms")
println("   Memory: $(round(memory_stdlib, digits=2)) MB")

println("\n2. RunningStats get_covariance() [current implementation]:")
result_current = @benchmark get_covariance($estimator) evals=1
time_current = median(result_current.times) / 1e6
memory_current = result_current.memory / 1024^2
println("   Time: $(round(time_current, digits=2)) ms")
println("   Memory: $(round(memory_current, digits=2)) MB")

println("\n3. Manual broadcasting division test:")
# Test the suspected bottleneck: wc.M2 ./ (wc.n - 1)
result_broadcast = @benchmark $(estimator.M2) ./ $(estimator.n - 1) evals=1
time_broadcast = median(result_broadcast.times) / 1e6
memory_broadcast = result_broadcast.memory / 1024^2
println("   Time: $(round(time_broadcast, digits=2)) ms")
println("   Memory: $(round(memory_broadcast, digits=2)) MB")

println("\n4. Scalar multiplication test:")
# Test more efficient alternative: wc.M2 * (1 / (wc.n - 1))
factor = 1.0 / (estimator.n - 1)
result_scalar = @benchmark $(estimator.M2) * $factor evals=1
time_scalar = median(result_scalar.times) / 1e6
memory_scalar = result_scalar.memory / 1024^2
println("   Time: $(round(time_scalar, digits=2)) ms")
println("   Memory: $(round(memory_scalar, digits=2)) MB")

println("\nAnalysis:")
println("  Broadcasting overhead: $(round(time_broadcast / time_scalar, digits=2))x slower")
println("  Memory overhead: $(round(memory_broadcast / memory_scalar, digits=2))x more memory")

# Verify results are identical
cov1 = estimator.M2 ./ (estimator.n - 1)
cov2 = estimator.M2 * factor
max_diff = maximum(abs.(cov1 - cov2))
println("  Max difference: $(max_diff) (should be ~0)")

println("\n" * "=" ^ 60)
println("Is broadcasting the bottleneck? $(time_broadcast > time_scalar ? "YES" : "NO")")