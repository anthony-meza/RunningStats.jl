using Statistics, LinearAlgebra
include("src/welford_covariance.jl")

function test_batch_consistency()
    println("Testing consistency across different batch sizes with identical data...")
    
    # Generate fixed test data
    Random.seed!(12345)  # Fixed seed for reproducibility
    n_samples = 1000
    n_features = 5
    X = randn(n_samples, n_features)
    
    # Test different batch configurations on the SAME data
    batch_configs = [
        (1000, "Single large batch"),
        (500, "Two batches of 500"),
        (250, "Four batches of 250"),
        (100, "Ten batches of 100"),
        (50, "Twenty batches of 50"),
        (10, "100 batches of 10"),
        (1, "1000 single updates")
    ]
    
    results = []
    
    for (batch_size, description) in batch_configs
        wc = WelfordEstimate(n_features)
        
        if batch_size == 1
            # Single updates
            for i in 1:n_samples
                update_single!(wc, X[i, :])
            end
        else
            # Batch updates
            for i in 1:batch_size:n_samples
                end_idx = min(i + batch_size - 1, n_samples)
                batch_data = X[i:end_idx, :]
                update_batch!(wc, batch_data)
            end
        end
        
        stats = get_statistics(wc)
        push!(results, (description, stats))
        
        println("$description:")
        println("  Sample count: $(stats.count)")
        println("  Mean[1]: $(stats.mean[1])")
        println("  Cov[1,1]: $(stats.covariance[1,1])")
        println("  Cov[1,2]: $(stats.covariance[1,2])")
        println()
    end
    
    # Compare all results against the reference (single updates)
    reference = results[end][2]  # Single updates as reference
    println("=== Differences from single-update reference ===")
    
    for i in 1:(length(results)-1)
        desc, stats = results[i]
        mean_diff = maximum(abs.(stats.mean - reference.mean))
        cov_diff = maximum(abs.(stats.covariance - reference.covariance))
        
        println("$desc:")
        println("  Max mean difference: $(mean_diff)")
        println("  Max covariance difference: $(cov_diff)")
        
        if mean_diff > 1e-12 || cov_diff > 1e-12
            println("  ⚠️  SIGNIFICANT DIFFERENCE DETECTED!")
        else
            println("  ✅ Within numerical precision")
        end
        println()
    end
    
    # Also compare against Julia's built-in functions
    reference_mean = vec(mean(X, dims=1))
    reference_cov = cov(X)
    
    println("=== Differences from Julia built-in functions ===")
    for (desc, stats) in results
        mean_diff = maximum(abs.(stats.mean - reference_mean))
        cov_diff = maximum(abs.(stats.covariance - reference_cov))
        
        println("$desc:")
        println("  Max mean difference: $(mean_diff)")
        println("  Max covariance difference: $(cov_diff)")
        
        if mean_diff > 1e-12 || cov_diff > 1e-12
            println("  ⚠️  SIGNIFICANT DIFFERENCE FROM BUILT-IN!")
        else
            println("  ✅ Matches built-in functions")
        end
        println()
    end
    
    return results
end

using Random
test_batch_consistency()