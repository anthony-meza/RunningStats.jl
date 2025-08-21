# Examples

This page provides practical examples of using RunningStats.jl.

## Basic Usage

```julia
using RunningStats
using Random

# Create an estimator
estimator = WelfordEstimate()

# Simulate getting data in batches
Random.seed!(42)
batch1 = randn(100, 3)  # First batch: 100 rows, 3 columns
batch2 = randn(50, 3)   # Second batch: 50 more rows

# Process first batch
stats1 = update_batch!(estimator, batch1)
println("Processed $(stats1.count) samples")
println("Current mean: $(round.(stats1.mean, digits=2))")

# Process second batch - statistics automatically update
stats2 = update_batch!(estimator, batch2)
println("Now processed $(stats2.count) total samples")
println("Updated mean: $(round.(stats2.mean, digits=2))")
```

## Comparing with Standard Julia

```julia
using Statistics

# Generate test data
data = randn(1000, 3)

# Method 1: Standard Julia (requires storing all data)
julia_mean = mean(data, dims=1)[:]
julia_cov = cov(data)

# Method 2: RunningStats (streaming)
estimator = WelfordEstimate()
update_batch!(estimator, data)
running_stats = get_statistics(estimator)

# Compare results - should be identical
println("Mean difference: $(maximum(abs.(julia_mean - running_stats.mean)))")
println("Covariance difference: $(maximum(abs.(julia_cov - running_stats.covariance)))")
```

## Updating with New Data

```julia
# Suppose new data arrives later...
new_data = randn(500, 3)

# Standard Julia: Need to combine and recompute everything  
all_data = vcat(data, new_data)
julia_mean_updated = mean(all_data, dims=1)[:]
julia_cov_updated = cov(all_data)

# RunningStats: Just update the existing estimator
update_batch!(estimator, new_data)
updated_stats = get_statistics(estimator)

# Results are still identical, but RunningStats never stored the original data!
println("Updated mean difference: $(maximum(abs.(julia_mean_updated - updated_stats.mean)))")
```

## Merging Statistics

```julia
# Process data from different sources
source1 = WelfordEstimate()  
source2 = WelfordEstimate()

update_batch!(source1, randn(200, 4))
update_batch!(source2, randn(300, 4))

# Combine the results (two ways)
combined = merge_estimate(source1, source2)    # Creates new estimator
merge_estimate!(source1, source2)             # Modifies source1 in-place

stats = get_statistics(combined)
println("Combined statistics from $(stats.count) total samples")
```

## Parallel Processing

```julia
using Distributed

# Function to process a data partition
function process_partition(data_chunk)
    est = WelfordEstimate()
    update_batch!(est, data_chunk)
    return est
end

# Split data across workers (simulated)
data_partitions = [randn(250, 3) for _ in 1:4]
estimator_results = map(process_partition, data_partitions)

# Merge all results
final_estimator = reduce(merge_estimate, estimator_results)
final_stats = get_statistics(final_estimator)

println("Processed $(final_stats.count) samples in parallel")
```