using RunningStats
using Statistics
using Random
using BenchmarkTools

# Set seed for reproducibility
Random.seed!(42)

# Test with large dataset to match your performance numbers
n_samples, n_features = 20000, 200  # Adjust to match your data size
X = randn(n_samples, n_features)

println("Analyzing update_batch! vs cov() bottleneck")
println("Data size: $n_samples samples × $n_features features")
println("=" ^ 60)

println("1. Julia stdlib cov():")
result_stdlib = @benchmark cov($X) evals=1
time_stdlib = median(result_stdlib.times) / 1e6
memory_stdlib = result_stdlib.memory / 1024^2
println("   Time: $(round(time_stdlib, digits=2)) ms")
println("   Memory: $(round(memory_stdlib, digits=2)) MB")
println("   Allocations: $(result_stdlib.allocs)")

println("\n2. RunningStats update_batch!:")
result_rs = @benchmark update_batch!(estimator, $X) setup=(estimator = WelfordEstimate()) evals=1
time_rs = median(result_rs.times) / 1e6
memory_rs = result_rs.memory / 1024^2
println("   Time: $(round(time_rs, digits=2)) ms")
println("   Memory: $(round(memory_rs, digits=2)) MB")
println("   Allocations: $(result_rs.allocs)")

println("\n3. Breaking down update_batch! components:")

# Test individual components
println("   a) Data conversion:")
result_convert = @benchmark convert(Matrix{Float64}, $X) evals=1
time_convert = median(result_convert.times) / 1e6
memory_convert = result_convert.memory / 1024^2
println("      Time: $(round(time_convert, digits=2)) ms")
println("      Memory: $(round(memory_convert, digits=2)) MB")

X_converted = convert(Matrix{Float64}, X)

println("   b) Batch mean computation:")
result_mean = @benchmark vec(mean($X_converted, dims=1)) evals=1
time_mean = median(result_mean.times) / 1e6
memory_mean = result_mean.memory / 1024^2
println("      Time: $(round(time_mean, digits=2)) ms")
println("      Memory: $(round(memory_mean, digits=2)) MB")

# Set up for centering tests
estimator = WelfordEstimate()
RunningStats.initialize!(estimator, n_features)
batch_mean = vec(mean(X_converted, dims=1))
estimator.mean .= batch_mean

println("   c) Matrix centering operations:")
result_center = @benchmark (X_centered_old = $X_converted .- $batch_mean'; X_centered_new = $X_converted .- $batch_mean') evals=1
time_center = median(result_center.times) / 1e6
memory_center = result_center.memory / 1024^2
println("      Time: $(round(time_center, digits=2)) ms")
println("      Memory: $(round(memory_center, digits=2)) MB")

# Set up for matrix multiplication test
X_centered = X_converted .- batch_mean'

println("   d) Matrix multiplication (X' * X):")
result_matmul = @benchmark $X_centered' * $X_centered evals=1
time_matmul = median(result_matmul.times) / 1e6
memory_matmul = result_matmul.memory / 1024^2
println("      Time: $(round(time_matmul, digits=2)) ms")
println("      Memory: $(round(memory_matmul, digits=2)) MB")

println("   e) get_statistics() call:")
estimator = WelfordEstimate()
update_batch!(estimator, X)
result_stats = @benchmark get_statistics($estimator) evals=1
time_stats = median(result_stats.times) / 1e6
memory_stats = result_stats.memory / 1024^2
println("      Time: $(round(time_stats, digits=2)) ms")
println("      Memory: $(round(memory_stats, digits=2)) MB")

println("\nComponent analysis:")
total_components = time_convert + time_mean + time_center + time_matmul + time_stats
println("   Sum of components: $(round(total_components, digits=2)) ms")
println("   Actual update_batch!: $(round(time_rs, digits=2)) ms")
println("   Overhead: $(round(time_rs - total_components, digits=2)) ms")

println("\nPerformance comparison:")
println("   RunningStats vs stdlib: $(round(time_rs / time_stdlib, digits=2))x slower")
println("   Memory ratio: $(round(memory_rs / memory_stdlib, digits=2))x more memory")

println("\n" * "=" ^ 60)
println("Biggest bottlenecks identified above ☝️")