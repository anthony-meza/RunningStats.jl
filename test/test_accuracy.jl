using Statistics, LinearAlgebra
include("src/welford_covariance.jl")

function test_accuracy()
    println("Testing numerical accuracy of batch vs single updates...")
    
    # Generate test data
    n_samples = 1000
    n_features = 5
    X = randn(n_samples, n_features)
    
    # Method 1: Single updates (reference implementation)
    wc_single = WelfordEstimate(n_features)
    for i in 1:n_samples
        update_single!(wc_single, X[i, :])
    end
    
    # Method 2: Batch update
    wc_batch = WelfordEstimate(n_features)
    update_batch!(wc_batch, X)
    
    # Method 3: Reference calculation using built-in functions
    reference_mean = vec(mean(X, dims=1))
    reference_cov = cov(X)
    
    # Compare results
    stats_single = get_statistics(wc_single)
    stats_batch = get_statistics(wc_batch)
    
    println("\n=== Comparison Results ===")
    println("Single vs Reference:")
    println("  Mean error: $(maximum(abs.(stats_single.mean - reference_mean)))")
    println("  Cov error:  $(maximum(abs.(stats_single.covariance - reference_cov)))")
    
    println("\nBatch vs Reference:")
    println("  Mean error: $(maximum(abs.(stats_batch.mean - reference_mean)))")
    println("  Cov error:  $(maximum(abs.(stats_batch.covariance - reference_cov)))")
    
    println("\nSingle vs Batch:")
    println("  Mean error: $(maximum(abs.(stats_single.mean - stats_batch.mean)))")
    println("  Cov error:  $(maximum(abs.(stats_single.covariance - stats_batch.covariance)))")
    
    # Test with smaller batches
    println("\n=== Testing with multiple small batches ===")
    wc_multi = WelfordEstimate(n_features)
    batch_size = 100
    for i in 1:batch_size:n_samples
        end_idx = min(i + batch_size - 1, n_samples)
        batch_data = X[i:end_idx, :]
        update_batch!(wc_multi, batch_data)
    end
    
    stats_multi = get_statistics(wc_multi)
    println("Multi-batch vs Reference:")
    println("  Mean error: $(maximum(abs.(stats_multi.mean - reference_mean)))")
    println("  Cov error:  $(maximum(abs.(stats_multi.covariance - reference_cov)))")
    
    return (stats_single, stats_batch, stats_multi, reference_mean, reference_cov)
end

test_accuracy()