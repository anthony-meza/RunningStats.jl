# RunningStats.jl

[![Tests](https://github.com/anthony-meza/RunningStats.jl/workflows/CI/badge.svg)](https://github.com/anthony-meza/RunningStats.jl/actions)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://anthony-meza.github.io/RunningStats.jl/)

A Julia package for streaming statistics computation using Welford's algorithm and Chan's parallel merging. Compute means, covariances, correlations, and variances without storing historical data.

## Key Features

✅ **Memory-efficient**: O(p²) space complexity, independent of sample count  
✅ **Numerically stable**: Uses Welford's algorithm to avoid catastrophic cancellation  
✅ **Streaming updates**: Process data in batches or one sample at a time  
✅ **Parallel merging**: Combine statistics from independent data streams  
✅ **Exact results**: Mathematically identical to standard batch computations  

## Quick Start

```julia
using RunningStats

# Create estimator and process data in batches
estimator = WelfordEstimate()
update_batch!(estimator, randn(1000, 3))

# Get all statistics at once
stats = get_statistics(estimator)
println("Processed $(stats.count) samples")
println("Means: $(stats.mean)")
println("Correlations:")
display(stats.correlation)
```

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/anthony-meza/RunningStats.jl")
```

## Usage

### Basic Streaming Updates

```julia
using RunningStats

# Create estimator and process data incrementally
estimator = WelfordEstimate()

# Process data in batches (most common)
batch1 = randn(1000, 3)
stats = update_batch!(estimator, batch1)
println("Processed $(stats.count) samples, mean = $(round.(stats.mean, digits=3))")

# Continue with more data - no memory accumulation!
batch2 = randn(500, 3) 
update_batch!(estimator, batch2)
final_stats = get_statistics(estimator)
```

### Single Sample Updates

```julia
# Process one sample at a time
estimator = WelfordEstimate()
for i in 1:1000
    sample = randn(3)  # 3-dimensional sample
    update_single!(estimator, sample)
end
```

### Parallel/Distributed Computing

```julia
# Process data chunks independently, then merge
estimator1 = WelfordEstimate()
estimator2 = WelfordEstimate()

update_batch!(estimator1, chunk1)
update_batch!(estimator2, chunk2)

# Merge results (two approaches)
combined = merge_estimate(estimator1, estimator2)    # Creates new estimator
merge_estimate!(estimator1, estimator2)              # Merges into estimator1
```

### Available Statistics

```julia
stats = get_statistics(estimator)

# Access individual components
println("Sample count: $(stats.count)")
println("Means: $(stats.mean)")
println("Covariance matrix: $(stats.covariance)")
println("Correlation matrix: $(stats.correlation)")  
println("Variances: $(stats.variance)")

# Or get them individually
cov_matrix = get_covariance(estimator)
corr_matrix = get_correlation(estimator)
```

## Performance & Accuracy

### Numerical Accuracy
RunningStats produces results identical to standard Julia functions (within machine precision):

```julia
using Statistics
data = randn(10000, 5)

# Standard approach 
julia_mean = mean(data, dims=1)[:]
julia_cov = cov(data)

# Streaming approach
estimator = WelfordEstimate()
update_batch!(estimator, data)
stats = get_statistics(estimator)

# Verify identical results
@assert maximum(abs.(julia_mean - stats.mean)) < 1e-14
@assert maximum(abs.(julia_cov - stats.covariance)) < 1e-14
```

### Complexity
- **Space**: O(p²) where p = number of features  
- **Time**: O(p²) per sample or batch update
- **Memory**: Independent of total sample count

### When to Use RunningStats

✅ **Use when:**
- Processing large datasets that don't fit in memory
- Data arrives in streams/batches over time
- Need to combine statistics from multiple sources
- Want to avoid storing raw data for privacy/storage reasons

❌ **Standard Julia may be better when:**
- Small datasets that easily fit in memory
- One-time batch computation
- Need only basic statistics (mean/std of single variables)

## API Reference

### Types

- `WelfordEstimate{T<:AbstractFloat}`: Main type for running statistical estimates

### Core Functions

- `WelfordEstimate()`: Create new estimator (defaults to Float64)
- `WelfordEstimate{Float32}()`: Create with specific precision
- `update_batch!(estimator, data_matrix)`: Process matrix data (samples × features)
- `update_single!(estimator, sample_vector)`: Process single sample
- `get_statistics(estimator)`: Returns (count, mean, covariance, correlation, variance)

### Statistics Access

- `get_covariance(estimator; corrected=true)`: Get covariance matrix
- `get_correlation(estimator)`: Get correlation matrix

### Merging Functions

- `merge_estimate(est1, est2)`: Combine estimators (creates new instance)
- `merge_estimate!(est1, est2)`: Merge est2 into est1 (modifies est1)

### Options

- `corrected::Bool`: Use sample (n-1) vs population (n) covariance (default: true)

## Algorithm Details

For detailed mathematical background including Welford's recursion formulas and Chan's parallel merging algorithm, see the [mathematical documentation](https://anthony-meza.github.io/RunningStats.jl/stable/mathematical_background/).

**Key insight**: Traditional variance computation σ² = E[X²] - (E[X])² suffers from catastrophic cancellation when the terms are nearly equal. Welford's algorithm avoids this by computing deviations from the running mean, maintaining numerical stability even for ill-conditioned data.

## References

1. Welford, B. P. (1962). "Note on a method for calculating corrected sums of squares and products." *Technometrics*, 4(3), 419-420.

2. Chan, T. F., Golub, G. H., & LeVeque, R. J. (1983). "Algorithms for computing the sample variance: Analysis and recommendations." *The American Statistician*, 37(3), 242-247.

3. Knuth, D. E. (1998). *The Art of Computer Programming, Volume 2: Seminumerical Algorithms* (3rd ed.). Addison-Wesley.

## License

MIT License - see [LICENSE](LICENSE) file for details.
