# API Reference

This page provides a comprehensive reference for all functions and types in RunningStats.jl.

## Types

### `WelfordEstimate{T<:AbstractFloat}`

Main type for streaming statistical estimates using Welford's algorithm.

**Fields:**
- `n::Int`: Number of samples processed
- `mean::Vector{T}`: Current mean vector
- `M2::Matrix{T}`: Sum of squared deviations matrix  
- `n_features::Int`: Number of features/dimensions

**Constructors:**
- `WelfordEstimate()`: Create with Float64 precision
- `WelfordEstimate{Float32}()`: Create with Float32 precision
- `WelfordEstimate{T}(n_features)`: Pre-allocate for known feature count

## Core Functions

### `initialize!(estimator, n_features)`
Initialize estimator for data with known number of features.

### `update_batch!(estimator, data_matrix)`
Update statistics with a batch of data points. Matrix should be samples × features.

**Returns:** Current statistics tuple `(count, mean, covariance, correlation, variance)`

### `update_single!(estimator, sample_vector)`
Update statistics with a single sample vector.

### `get_statistics(estimator)`
Get all current statistics.

**Returns:** Named tuple with fields:
- `count`: Number of samples processed
- `mean`: Mean vector
- `covariance`: Covariance matrix
- `correlation`: Correlation matrix  
- `variance`: Variance vector (diagonal of covariance)

### `get_covariance(estimator; corrected=true)`
Get the current covariance matrix.

**Parameters:**
- `corrected::Bool`: Use sample covariance (n-1) vs population (n). Default: true

### `get_correlation(estimator)`
Get the current correlation matrix (standardized covariance).

## Merging Functions

### `merge_estimate(est1, est2)`
Combine two estimators into a new estimator (non-destructive).

**Returns:** New `WelfordEstimate` containing merged statistics

### `merge_estimate!(est1, est2)`
Merge `est2` into `est1` (modifies `est1` in-place).

**Returns:** Modified `est1`