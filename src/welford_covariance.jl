# WelfordEstimate - Streaming statistics estimator using Welford's algorithm
# Computes means, covariances, correlations, and variances without storing historical samples.

mutable struct WelfordEstimate{T<:AbstractFloat}
    n::Int
    mean::Vector{T}
    M2::Matrix{T}
    n_features::Int
    
    function WelfordEstimate{T}(n_features::Int=0) where T<:AbstractFloat
        new{T}(0, Vector{T}(), Matrix{T}(undef, 0, 0), n_features)
    end
end

WelfordEstimate(n_features::Int=0) = WelfordEstimate{Float64}(n_features)

# Initialize the estimator for data with a known number of features
function initialize!(wc::WelfordEstimate{T}, n_features::Int) where T
    wc.n_features = n_features
    wc.mean = zeros(T, n_features)
    wc.M2 = zeros(T, n_features, n_features)
    return wc
end

# Update statistics with a batch of data points (most common function)
function update_batch!(wc::WelfordEstimate{T}, X::AbstractMatrix{<:Real}) where T
    X_converted = convert(Matrix{T}, X)
    n_samples, n_features = size(X_converted)
    
    if wc.n == 0
        initialize!(wc, n_features)
    elseif n_features != wc.n_features
        throw(DimensionMismatch("Expected $(wc.n_features) features, got $n_features"))
    end
    
    for i in 1:n_samples
        sample = view(X_converted, i, :)
        update_single!(wc, sample)
    end
    
    return get_statistics(wc)
end

# Update statistics with a single sample (vector input)
function update_batch!(wc::WelfordEstimate{T}, X::AbstractVector{<:Real}) where T
    X_converted = convert(Vector{T}, X)
    return update_batch!(wc, reshape(X_converted, 1, :))
end

# Update statistics with a single data point
function update_single!(wc::WelfordEstimate{T}, x::AbstractVector) where T
    wc.n += 1
    
    delta = x - wc.mean
    wc.mean .+= delta ./ wc.n
    
    delta2 = x - wc.mean
    wc.M2 .+= delta * delta2'
    
    return nothing
end

# Get the current covariance matrix
function get_covariance(wc::WelfordEstimate{T}; corrected::Bool=true) where T
    if wc.n == 0
        return Matrix{T}(undef, 0, 0)
    end
    
    ddof = corrected ? 1 : 0
    if wc.n <= ddof
        return fill(T(NaN), wc.n_features, wc.n_features)
    end
    
    return wc.M2 ./ (wc.n - ddof)
end

# Get the current correlation matrix (standardized covariance)
function get_correlation(wc::WelfordEstimate{T}) where T
    cov_matrix = get_covariance(wc)
    std_devs = sqrt.(diag(cov_matrix))
    
    std_devs = replace(x -> x == 0 ? one(T) : x, std_devs)
    
    return cov_matrix ./ (std_devs * std_devs')
end

# Get all computed statistics in one convenient package
function get_statistics(wc::WelfordEstimate{T}) where T
    if wc.n == 0
        return (count=0, mean=Vector{T}(), covariance=Matrix{T}(undef, 0, 0), 
                correlation=Matrix{T}(undef, 0, 0), variance=Vector{T}())
    end
    
    cov_matrix = get_covariance(wc)
    return (
        count=wc.n,
        mean=copy(wc.mean),
        covariance=cov_matrix,
        correlation=get_correlation(wc),
        variance=diag(cov_matrix)
    )
end

# Merge statistics from two estimators, modifying the first one
function merge_estimate!(wc1::WelfordEstimate{T}, wc2::WelfordEstimate{T}) where T
    if wc2.n == 0
        return wc1
    end
    
    if wc1.n == 0
        wc1.n = wc2.n
        wc1.mean = copy(wc2.mean)
        wc1.M2 = copy(wc2.M2)
        wc1.n_features = wc2.n_features
        return wc1
    end
    
    if wc1.n_features != wc2.n_features
        throw(DimensionMismatch("Cannot merge WelfordEstimate instances with different numbers of features"))
    end
    
    combined_n = wc1.n + wc2.n
    
    delta = wc2.mean - wc1.mean
    combined_mean = wc1.mean + delta * wc2.n / combined_n
    
    combined_M2 = wc1.M2 + wc2.M2 + (delta * delta') * (wc1.n * wc2.n / combined_n)
    
    wc1.n = combined_n
    wc1.mean = combined_mean
    wc1.M2 = combined_M2
    
    return wc1
end

# Create a new estimator by merging two existing ones (non-destructive)
function merge_estimate(wc1::WelfordEstimate{T}, wc2::WelfordEstimate{T}) where T
    result = WelfordEstimate{T}()
    if wc1.n > 0
        result.n = wc1.n
        result.mean = copy(wc1.mean)
        result.M2 = copy(wc1.M2)
        result.n_features = wc1.n_features
    end
    merge_estimate!(result, wc2)
    return result
end