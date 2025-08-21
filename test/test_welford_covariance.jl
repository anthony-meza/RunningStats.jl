@testset "WelfordEstimate Tests" begin
    
    @testset "Welford vs Standard Covariance vs True Covariance" begin
        Random.seed!(42)
        
        # Define true population parameters
        true_mean = [1.0, 2.0, 3.0]
        true_cov = [2.0 0.5 -0.3;
                    0.5 1.5 0.2;
                    -0.3 0.2 1.0]
        
        # Generate multivariate normal data with known parameters
        using Distributions
        dist = MvNormal(true_mean, true_cov)
        n_samples = 10_000
        data = rand(dist, n_samples)'  # Transpose to get (n_samples, n_features)
        
        # Compute statistics using Welford's streaming algorithm with multiple batches
        wc = WelfordEstimate{Float64}()
        batch_size = 200
        for i in 1:batch_size:n_samples
            end_idx = min(i + batch_size - 1, n_samples)
            batch = data[i:end_idx, :]
            update_batch!(wc, batch)
        end
        welford_stats = get_statistics(wc)
        
        # Compute statistics using standard Julia functions on all data at once
        standard_mean = mean(data, dims=1)[:]
        standard_cov = cov(data)
        
        # Test 1: Streaming batch updates should match full-data computation exactly
        @test welford_stats.mean ≈ standard_mean atol=1e-12
        @test welford_stats.covariance ≈ standard_cov atol=1e-12
        # Test 2: Both estimates should approximate true population parameters
        @test welford_stats.mean ≈ true_mean atol=0.1
        @test welford_stats.covariance ≈ true_cov atol=0.1

        # Pretty print comparison
        println("\n" * "="^60)
        println("COVARIANCE COMPARISON")
        println("="^60)
        
        println("\nDiagonal variances:")
        println("  True:     $(round.(diag(true_cov), digits=3))")
        println("  Standard: $(round.(diag(standard_cov), digits=3))")
        println("  Welford:  $(round.(diag(welford_stats.covariance), digits=3))")
        
        println("\nMeans:")
        println("  True:     $(round.(true_mean, digits=3))")
        println("  Standard: $(round.(standard_mean, digits=3))")
        println("  Welford:  $(round.(welford_stats.mean, digits=3))")
        
        println("\nFull Covariance Matrices:")
        println("  True:")
        display(round.(true_cov, digits=3))
        println("\n  Standard:")
        display(round.(standard_cov, digits=3))
        println("\n  Welford:")
        display(round.(welford_stats.covariance, digits=3))
        
        println("\nDifferences (Welford - Standard):")
        mean_diff = welford_stats.mean - standard_mean
        cov_diff = welford_stats.covariance - standard_cov
        println("  Mean diff:    $(mean_diff)")
        println("  Cov diag diff: $(diag(cov_diff))")
        println("  Max cov diff:  $(maximum(abs.(cov_diff)))")
        
        println("="^60)

    end
    

end