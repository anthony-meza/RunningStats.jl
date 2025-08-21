using Statistics, LinearAlgebra, Random
include("src/welford_covariance.jl")

function test_many_batch_accumulation()
    println("Testing numerical stability with many accumulated batches...")
    
    Random.seed!(12345)
    n_features = 5
    batch_size = 100
    
    # Test scenarios with increasing number of batches
    batch_counts = [10, 100, 1000, 10000]
    
    for n_batches in batch_counts
        println("\n=== Testing $n_batches batches of size $batch_size ===")
        
        # Generate all data upfront for reference
        total_samples = n_batches * batch_size
        X_all = randn(total_samples, n_features)
        
        # Method 1: Process all data at once (reference)
        wc_ref = WelfordEstimate(n_features)
        update_batch!(wc_ref, X_all)
        stats_ref = get_statistics(wc_ref)
        
        # Method 2: Process in many small batches
        wc_accum = WelfordEstimate(n_features)
        for i in 1:n_batches
            start_idx = (i-1) * batch_size + 1
            end_idx = i * batch_size
            batch_data = X_all[start_idx:end_idx, :]
            update_batch!(wc_accum, batch_data)
            
            # Print progress for large tests
            if n_batches >= 1000 && i % 1000 == 0
                println("  Processed batch $i/$n_batches")
            end
        end
        stats_accum = get_statistics(wc_accum)
        
        # Method 3: Built-in Julia functions (ground truth)
        ref_mean = vec(mean(X_all, dims=1))
        ref_cov = cov(X_all)
        
        # Compare results
        mean_diff_ref = maximum(abs.(stats_accum.mean - stats_ref.mean))
        cov_diff_ref = maximum(abs.(stats_accum.covariance - stats_ref.covariance))
        
        mean_diff_julia = maximum(abs.(stats_accum.mean - ref_mean))
        cov_diff_julia = maximum(abs.(stats_accum.covariance - ref_cov))
        
        println("Total samples processed: $(total_samples)")
        println("Accumulated vs single batch:")
        println("  Max mean difference: $(mean_diff_ref)")
        println("  Max covariance difference: $(cov_diff_ref)")
        
        println("Accumulated vs Julia built-in:")
        println("  Max mean difference: $(mean_diff_julia)")
        println("  Max covariance difference: $(cov_diff_julia)")
        
        # Check for numerical degradation
        tolerance = 1e-10  # Much stricter than machine precision
        if mean_diff_ref > tolerance || cov_diff_ref > tolerance
            println("  ⚠️  NUMERICAL DEGRADATION DETECTED!")
        else
            println("  ✅ No numerical degradation")
        end
        
        # Check final statistics
        println("Final mean[1]: $(stats_accum.mean[1])")
        println("Final cov[1,1]: $(stats_accum.covariance[1,1])")
    end
    
    println("\n=== Testing with extreme accumulation (100,000 tiny batches) ===")
    wc_extreme = WelfordEstimate(3)  # Smaller for speed
    
    n_extreme_batches = 100000
    extreme_batch_size = 5
    
    Random.seed!(12345)
    
    for i in 1:n_extreme_batches
        tiny_batch = randn(extreme_batch_size, 3)
        update_batch!(wc_extreme, tiny_batch)
        
        if i % 10000 == 0
            println("  Processed $i/$n_extreme_batches batches")
        end
    end
    
    stats_extreme = get_statistics(wc_extreme)
    println("After $(n_extreme_batches) tiny batches:")
    println("  Total samples: $(stats_extreme.count)")
    println("  Mean: $(stats_extreme.mean)")
    println("  Cov diagonal: $(diag(stats_extreme.covariance))")
    
    return stats_extreme
end

test_many_batch_accumulation()