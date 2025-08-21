# Mathematical Background

This page provides the mathematical foundation for the algorithms implemented in RunningStats.jl.

## The Streaming Statistics Problem

### Traditional vs Online Approaches

**Traditional approach:**
$$\text{Data: } x_1, x_2, \ldots, x_n \rightarrow \text{Store all} \rightarrow \text{Compute } \mu = \frac{1}{n}\sum_{i=1}^n x_i, \quad \Sigma = \frac{1}{n-1}\sum_{i=1}^n (x_i-\mu)(x_i-\mu)^T$$

**Online approach:**
$$x_1 \rightarrow \text{Update estimates} \rightarrow x_2 \rightarrow \text{Update estimates} \rightarrow \cdots \rightarrow \text{Final } \mu, \Sigma$$

The online approach uses O(p²) memory for the algorithm state plus O(batch_size × p) for processing batches, where p is the number of features. The key advantage is that memory usage doesn't grow with the total number of samples processed over time.

## Welford's Algorithm (1962)

### Mathematical Formulation

For a sequence of $p$-dimensional observations $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n$, Welford's algorithm maintains:

- $n$: running count of observations
- $\boldsymbol{\mu}$: running mean vector $(p \times 1)$  
- $\mathbf{M}_2$: running sum of outer products matrix $(p \times p)$

### Update Rules

For each new observation $\mathbf{x}_k$:

$$n \leftarrow n + 1$$
$$\boldsymbol{\delta} = \mathbf{x}_k - \boldsymbol{\mu}$$
$$\boldsymbol{\mu} \leftarrow \boldsymbol{\mu} + \frac{\boldsymbol{\delta}}{n}$$
$$\mathbf{M}_2 \leftarrow \mathbf{M}_2 + \boldsymbol{\delta}(\mathbf{x}_k - \boldsymbol{\mu})^T$$

### Final Estimates

From the accumulated statistics $(n, \boldsymbol{\mu}, \mathbf{M}_2)$, we can compute:

- **Sample covariance**: $\boldsymbol{\Sigma} = \frac{\mathbf{M}_2}{n-1}$ (Bessel's correction)
- **Population covariance**: $\boldsymbol{\Sigma}_{\text{pop}} = \frac{\mathbf{M}_2}{n}$ 
- **Sample correlation**: $\mathbf{R} = \mathbf{D}^{-1}\boldsymbol{\Sigma}\mathbf{D}^{-1}$ where $\mathbf{D} = \text{diag}(\sqrt{\sigma_{11}}, \sqrt{\sigma_{22}}, \ldots)$
- **Individual variances**: $\text{diag}(\boldsymbol{\Sigma}) = (\sigma_1^2, \ldots, \sigma_p^2)$

### Edge Cases

- **Empty estimator** ($n = 0$): Returns empty matrices/vectors
- **Single sample** ($n = 1$): Sample covariance is undefined (NaN), population covariance is zero matrix
- **Zero variance**: Correlation computation handles $\sigma_{ii} = 0$ by treating as $\sigma_{ii} = 1$ to avoid division by zero

### Numerical Stability

Traditional "textbook" variance formula $\sigma^2 = E[X^2] - (E[X])^2$ suffers from catastrophic cancellation when $E[X^2] \approx (E[X])^2$. Welford's algorithm avoids this by:

1. Computing deviations from running mean rather than raw second moments
2. Using the mathematically equivalent but numerically stable recurrence relation

**Stability comparison:**
- Naive: $\sigma^2 = \frac{\sum x^2}{n} - \left(\frac{\sum x}{n}\right)^2$ → **susceptible to cancellation**  
- Welford: $\sigma^2 = \frac{M_2}{n-1}$ where $M_2$ accumulates $(x-\mu)^2$ → **numerically stable**

## Chan's Parallel Algorithm (1983)

### The Merging Problem

Given two sets of statistics computed independently:
- Stream A: $(n_A, \boldsymbol{\mu}_A, \mathbf{M}_{2A})$ from data $\{x_1, x_2, \ldots, x_{n_A}\}$
- Stream B: $(n_B, \boldsymbol{\mu}_B, \mathbf{M}_{2B})$ from data $\{y_1, y_2, \ldots, y_{n_B}\}$

Find the equivalent statistics for the combined dataset $\{x_1, \ldots, x_{n_A}, y_1, \ldots, y_{n_B}\}$.

### Chan's Solution

The exact merged statistics are computed as:

$$n_{\text{combined}} = n_A + n_B$$
$$\boldsymbol{\delta} = \boldsymbol{\mu}_B - \boldsymbol{\mu}_A$$
$$\boldsymbol{\mu}_{\text{combined}} = \boldsymbol{\mu}_A + \boldsymbol{\delta}\frac{n_B}{n_{\text{combined}}}$$
$$\mathbf{M}_{2,\text{combined}} = \mathbf{M}_{2A} + \mathbf{M}_{2B} + \boldsymbol{\delta}\boldsymbol{\delta}^T\frac{n_A n_B}{n_{\text{combined}}}$$

### Algorithm Implementation

The merging process handles several edge cases:

1. **Empty source** ($n_B = 0$): Result equals target ($n_A, \boldsymbol{\mu}_A, \mathbf{M}_{2A}$)
2. **Empty target** ($n_A = 0$): Target becomes copy of source ($n_B, \boldsymbol{\mu}_B, \mathbf{M}_{2B}$)
3. **Dimension mismatch**: Throws `DimensionMismatch` if feature counts differ
4. **Numerical precision**: All operations maintain the numerical stability of underlying Welford updates

### Properties

- **Exactness**: Merged result is identical to processing all data together
- **Associativity**: Order of merging doesn't matter
- **Numerical stability**: Inherits stability properties from underlying Welford updates

### Applications

- **Map-reduce**: Process data chunks on different machines, merge results
- **Hierarchical computation**: Combine statistics from organizational units
- **Streaming aggregation**: Merge statistics from different time windows

## Complexity Analysis

### Memory Complexity
- **Algorithm state**: $O(p^2)$ where $p$ is the number of features
- **Single updates**: $O(p^2)$ total (only stores current state)
- **Batch updates**: $O(p^2 + \text{batch\_size} \times p)$ during processing
- **Key advantage**: Memory doesn't grow with total number of samples over time

### Time Complexity
- **Per sample**: $O(p^2)$ for Welford update
- **Per batch**: $O(\text{batch\_size} \times p^2)$ 
- **Merging**: $O(p^2)$ to combine two estimators

### Numerical Stability
- Uses Welford's algorithm for stable computation
- Avoids catastrophic cancellation in variance computations
- Suitable for streaming applications with millions of samples

## References

1. Welford, B. P. (1962). "Note on a method for calculating corrected sums of squares and products." *Technometrics*, 4(3), 419-420.

2. Chan, T. F., Golub, G. H., & LeVeque, R. J. (1983). "Algorithms for computing the sample variance: Analysis and recommendations." *The American Statistician*, 37(3), 242-247.

3. Knuth, D. E. (1998). *The Art of Computer Programming, Volume 2: Seminumerical Algorithms* (3rd ed.). Addison-Wesley.