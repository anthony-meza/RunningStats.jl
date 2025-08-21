# RunningStats.jl

A Julia package for computing statistics on streaming data using numerically stable online algorithms.

## Mathematical Background

Many statistical computations require storing entire datasets in memory, which becomes problematic for large-scale data analysis. This package implements **online algorithms** that update statistics incrementally as new data arrives, using only constant memory regardless of dataset size.

### The Streaming Statistics Problem

Traditional approach:
$$\text{Data: } x_1, x_2, \ldots, x_n \rightarrow \text{Store all} \rightarrow \text{Compute } \mu = \frac{1}{n}\sum_{i=1}^n x_i, \quad \Sigma = \frac{1}{n-1}\sum_{i=1}^n (x_i-\mu)(x_i-\mu)^T$$

Online approach:
$$x_1 \rightarrow \text{Update estimates} \rightarrow x_2 \rightarrow \text{Update estimates} \rightarrow \cdots \rightarrow \text{Final } \mu, \Sigma$$

### Welford's Algorithm (1962)

This package implements Welford's algorithm for numerically stable online computation of sample statistics. For each new observation $\mathbf{x}$, the algorithm updates:

1. **Sample count**: $n \leftarrow n + 1$
2. **Mean**: $\boldsymbol{\mu} \leftarrow \boldsymbol{\mu} + \frac{\mathbf{x} - \boldsymbol{\mu}}{n}$
3. **Sum of squared deviations**: $\mathbf{M_2} \leftarrow \mathbf{M_2} +$
4. **Sample covariance**: $\boldsymbol{\Sigma} = \frac{\mathbf{M_2}}{n-1}$

### Chan's Parallel Algorithm (1983)

For parallel and distributed computing scenarios, we implement Chan, Golub, and LeVeque's algorithm for combining statistics from independent data streams. This solves the problem: *"How do we merge statistics computed separately on different data partitions?"*

**The merging problem**: Given two sets of statistics computed independently:
- Stream A: $(n_A, \boldsymbol{\mu}_A, \mathbf{M}_{2A})$ from data $\{x_1, x_2, \ldots, x_{n_A}\}$
- Stream B: $(n_B, \boldsymbol{\mu}_B, \mathbf{M}_{2B})$ from data $\{y_1, y_2, \ldots, y_{n_B}\}$

Find the equivalent statistics for the combined dataset $\{x_1, \ldots, x_{n_A}, y_1, \ldots, y_{n_B}\}$.

**Chan's solution** provides the exact merged statistics:
$$n_{\text{combined}} = n_A + n_B$$
$$\boldsymbol{\delta} = \boldsymbol{\mu}_B - \boldsymbol{\mu}_A$$
$$\boldsymbol{\mu}_{\text{combined}} = \boldsymbol{\mu}_A + \boldsymbol{\delta}\frac{n_B}{n_{\text{combined}}}$$
$$\mathbf{M}_{2,\text{combined}} = \mathbf{M}_{2A} + \mathbf{M}_{2B} + \boldsymbol{\delta}\boldsymbol{\delta}^T\frac{n_A n_B}{n_{\text{combined}}}$$

**Applications**:
- **Map-reduce**: Process data chunks on different machines, merge results
- **Hierarchical computation**: Combine statistics from organizational units
- **Streaming aggregation**: Merge statistics from different time windows

**Key advantages**:
- **Exactness**: Merged result is identical to processing all data together
- **Numerical stability**: Inherits stability properties from underlying Welford updates
- **Associativity**: Order of merging doesn't matter: merge(A, merge(B, C)) = merge(merge(A, B), C)
- **Memory efficiency**: $O(p^2)$ space for $p$-dimensional data, independent of sample size
- **Single pass**: No need to revisit previous observations

## Installation

```julia
using Pkg
Pkg.add("RunningStats")
```

## 5-Minute Quick Start

```julia
using RunningStats
using Random

# Create your statistics tracker
estimator = WelfordEstimate()

# Simulate getting data in batches (like from a file or network)
Random.seed!(42)
batch1 = randn(100, 3)  # First batch: 100 rows, 3 columns
batch2 = randn(50, 3)   # Second batch: 50 more rows

# Process first batch - no need to store all the data!
stats1 = update_batch!(estimator, batch1)
println("✅ Processed $(stats1.count) samples")
println("📊 Current average: $(round.(stats1.mean, digits=2))")

# Process second batch - statistics automatically update
stats2 = update_batch!(estimator, batch2)
println("✅ Now processed $(stats2.count) total samples")
println("📊 Updated average: $(round.(stats2.mean, digits=2))")

# Get the final correlation matrix (how variables relate to each other)
correlations = get_correlation(estimator)
println("🔗 Correlations between variables:")
display(round.(correlations, digits=2))
```

## How to Use It

### The Main Tool: WelfordEstimate

Think of `WelfordEstimate` as a smart calculator that remembers everything but doesn't hog your memory:

```julia
# Create your calculator
estimator = WelfordEstimate()               # Simple creation
estimator = WelfordEstimate{Float32}()      # If you want to save even more memory

# Feed it data (any of these work!)
update_batch!(estimator, your_data_matrix)  # Most common: feed a matrix
update_batch!(estimator, single_row)        # Or just one row at a time
update_single!(estimator, single_sample)    # Alternative way for single samples

# Get your results
all_stats = get_statistics(estimator)       # Everything in one go
covariances = get_covariance(estimator)     # Just the covariance matrix  
correlations = get_correlation(estimator)   # Just the correlation matrix
```

### What You Get Back

When you call `get_statistics()`, you get everything you need:

- **`count`**: How many data points you've processed so far
- **`mean`**: The average of each variable (column)
- **`covariance`**: How variables change together (useful for understanding relationships)
- **`correlation`**: Standardized relationships between variables (-1 to +1 scale)
- **`variance`**: How spread out each variable is

### Combining Results from Different Sources

Got data coming from multiple sources? No problem! You can combine the statistics:

```julia
# Say you have two different data sources
source1 = WelfordEstimate()
source2 = WelfordEstimate()

# Process them separately
update_batch!(source1, data_from_source_1)
update_batch!(source2, data_from_source_2)

# Combine the results (two ways to do it)
combined = merge_estimate(source1, source2)  # Creates a new combined estimator
merge_estimate!(source1, source2)            # Adds source2's data into source1
```

## Examples

### Comparing with Standard Julia Functions

```julia
using Statistics

# Generate initial test data
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
# Should be ~1e-15 (machine precision)
```

### Updating with New Data

```julia
# Now suppose new data arrives...
new_data = randn(500, 3)

# Standard Julia: Need to combine datasets and recompute everything
all_data = vcat(data, new_data)
julia_mean_updated = mean(all_data, dims=1)[:]
julia_cov_updated = cov(all_data)

# RunningStats: Just update the existing estimator
update_batch!(estimator, new_data)
updated_stats = get_statistics(estimator)

# Results are still identical
println("Updated mean difference: $(maximum(abs.(julia_mean_updated - updated_stats.mean)))")
println("Updated covariance difference: $(maximum(abs.(julia_cov_updated - updated_stats.covariance)))")

# But RunningStats never stored the original data!
println("Processed $(updated_stats.count) total samples without storing any raw data")
```

## Algorithm Details

### Welford's Online Algorithm

For a sequence of $p$-dimensional observations $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n$, Welford's algorithm maintains:

- $n$: running count of observations
- $\boldsymbol{\mu}$: running mean vector $(p \times 1)$  
- $\mathbf{M_2}$: running sum of outer products matrix $(p \times p)$

**Update rules for new observation $\mathbf{x}_k$**:
$$n \leftarrow n + 1$$
$$\boldsymbol{\delta} = \mathbf{x}_k - \boldsymbol{\mu}$$
$$\boldsymbol{\mu} \leftarrow \boldsymbol{\mu} + \frac{\boldsymbol{\delta}}{n}$$
$$\mathbf{M_2} \leftarrow \mathbf{M_2} + \boldsymbol{\delta}(\mathbf{x}_k - \boldsymbol{\mu})^T$$

**Final estimates**:
- Sample covariance: $\boldsymbol{\Sigma} = \frac{\mathbf{M_2}}{n-1}$
- Sample correlation: $\mathbf{R} = \mathbf{D}^{-1}\boldsymbol{\Sigma}\mathbf{D}^{-1}$ where $\mathbf{D} = \text{diag}(\sqrt{\sigma_{11}}, \sqrt{\sigma_{22}}, \ldots)$

### Parallel Merging (Chan's Algorithm)

To combine statistics from two streams with $(n_1, \boldsymbol{\mu}_1, \mathbf{M}_2^{(1)})$ and $(n_2, \boldsymbol{\mu}_2, \mathbf{M}_2^{(2)})$:

$$n = n_1 + n_2$$
$$\boldsymbol{\delta} = \boldsymbol{\mu}_2 - \boldsymbol{\mu}_1$$
$$\boldsymbol{\mu} = \boldsymbol{\mu}_1 + \boldsymbol{\delta}\frac{n_2}{n}$$
$$\mathbf{M}_2 = \mathbf{M}_2^{(1)} + \mathbf{M}_2^{(2)} + \boldsymbol{\delta}\boldsymbol{\delta}^T\frac{n_1 n_2}{n}$$

This allows **embarrassingly parallel** computation: process data chunks independently, then merge results.

### Numerical Stability

Traditional "textbook" variance formula $\sigma^2 = E[X^2] - (E[X])^2$ suffers from catastrophic cancellation when $E[X^2] \approx (E[X])^2$. Welford's algorithm avoids this by:

1. Computing deviations from running mean rather than raw second moments
2. Using the mathematically equivalent but numerically stable recurrence relation

**Stability comparison**:
- Naive: $\sigma^2 = \frac{\sum x^2}{n} - \left(\frac{\sum x}{n}\right)^2$ → **susceptible to cancellation**  
- Welford: $\sigma^2 = \frac{M_2}{n-1}$ where $M_2$ accumulates $(x-\mu)^2$ → **numerically stable**

## Performance

- **Space Complexity**: $O(p^2)$ where $p$ is the number of features
- **Time Complexity**: $O(p^2)$ per sample update  
- **Numerical Stability**: Uses Welford's algorithm for stable computation
- **Memory**: No need to store historical data points

## API Reference

### Types

- `WelfordEstimate{T<:AbstractFloat}`: Main type for running statistical estimates

### Functions

- `initialize!(wc, n_features)`: Initialize with known feature count
- `update_batch!(wc, X)`: Update with batch of data
- `update_single!(wc, x)`: Update with single sample
- `get_statistics(wc)`: Get all current statistics
- `get_covariance(wc; corrected=true)`: Get covariance matrix
- `get_correlation(wc)`: Get correlation matrix
- `merge_estimate!(wc1, wc2)`: Merge wc2 into wc1 (in-place)
- `merge_estimate(wc1, wc2)`: Create new merged instance

### Options

- `corrected::Bool`: Use sample (n-1) vs population (n) covariance (default: true)

## References

1. Welford, B. P. (1962). "Note on a method for calculating corrected sums of squares and products." *Technometrics*, 4(3), 419-420.

2. Chan, T. F., Golub, G. H., & LeVeque, R. J. (1983). "Algorithms for computing the sample variance: Analysis and recommendations." *The American Statistician*, 37(3), 242-247.

3. Knuth, D. E. (1998). *The Art of Computer Programming, Volume 2: Seminumerical Algorithms* (3rd ed.). Addison-Wesley.

## License

MIT License
