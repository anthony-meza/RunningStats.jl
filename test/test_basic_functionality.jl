@testset "RunningStats Tests" begin

    @testset "Basic Functionality" begin
        # Test empty initialization
        wc = WelfordEstimate()
        @test wc.n == 0
        empty_stats = get_statistics(wc)
        @test empty_stats.count == 0
        
        # Test single sample
        wc = WelfordEstimate()
        update_batch!(wc, [1.0, 2.0])
        stats = get_statistics(wc)
        @test stats.count == 1
        @test stats.mean ≈ [1.0, 2.0]
        @test all(isnan.(stats.covariance))  # Can't compute covariance with n=1
        
        # Test dimension mismatch error
        wc = WelfordEstimate()
        update_batch!(wc, randn(5, 3))
        @test_throws DimensionMismatch update_batch!(wc, randn(3, 2))
        
        # Test merge functionality
        Random.seed!(123)
        wc1 = WelfordEstimate()
        wc2 = WelfordEstimate()
        
        data1 = randn(50, 2)
        data2 = randn(30, 2)
        
        update_batch!(wc1, data1)
        update_batch!(wc2, data2)
        
        wc_merged = merge_estimate(wc1, wc2)
        wc_direct = WelfordEstimate()
        update_batch!(wc_direct, vcat(data1, data2))
        
        merged_stats = get_statistics(wc_merged)
        direct_stats = get_statistics(wc_direct)
        
        @test merged_stats.count == direct_stats.count
        @test merged_stats.mean ≈ direct_stats.mean atol=1e-12
        @test merged_stats.covariance ≈ direct_stats.covariance atol=1e-12
    end
end