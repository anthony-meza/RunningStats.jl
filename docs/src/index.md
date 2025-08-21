# RunningStats.jl

*Efficient computation of streaming statistics using numerically stable online algorithms.*

## Overview

RunningStats.jl provides tools for computing statistics on data streams without storing the entire dataset in memory. This is particularly useful for:

- Large datasets that don't fit in memory
- Real-time streaming applications  
- Parallel and distributed computing
- Memory-constrained environments

## Quick Start

```julia
using RunningStats

# Create an estimator
estimator = WelfordEstimate()

# Process data in chunks
update_batch!(estimator, randn(1000, 3))
update_batch!(estimator, randn(500, 3))

# Get results
stats = get_statistics(estimator)
println("Mean: ", stats.mean)
println("Covariance diagonal: ", stats.variance)
```

## Key Features

- **Memory Efficient**: O(p²) space complexity, independent of sample size
- **Numerically Stable**: Uses Welford's algorithm to avoid catastrophic cancellation
- **Parallel Friendly**: Implements Chan's algorithm for merging statistics
- **Type Flexible**: Supports different floating-point precisions

## Package Contents

```@contents
Pages = ["mathematical_background.md", "api.md", "examples.md"]
```