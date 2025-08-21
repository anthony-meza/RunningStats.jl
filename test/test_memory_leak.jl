using Pkg
using Statistics
using LinearAlgebra
Pkg.activate(".")

include("src/welford_covariance.jl")

function memory_usage_mb()
    GC.gc()  # Force garbage collection
    return Base.gc_live_bytes() / 1024 / 1024  # Convert to MB
end

function test_memory_leak()
    println("Testing memory usage with different batch configurations...")
    
    # Test parameters
    n_features = 100
    batch_size = 1000
    
    # Test 1: Single large batch
    println("\n=== Test 1: Single large batch ===")
    initial_memory = memory_usage_mb()
    println("Initial memory: $(Base.round(initial_memory, digits=2)) MB")
    
    wc1 = WelfordEstimate(n_features)
    X_large = randn(10000, n_features)  # 10k samples
    update_batch!(wc1, X_large)
    
    after_large_batch = memory_usage_mb()
    println("After large batch: $(Base.round(after_large_batch, digits=2)) MB")
    println("Memory increase: $(Base.round(after_large_batch - initial_memory, digits=2)) MB")
    
    # Test 2: Multiple small batches (same total samples)
    println("\n=== Test 2: Multiple small batches (same total) ===")
    GC.gc()
    before_small_batches = memory_usage_mb()
    println("Before small batches: $(Base.round(before_small_batches, digits=2)) MB")
    
    wc2 = WelfordEstimate(n_features)
    for i in 1:10
        X_small = randn(batch_size, n_features)  # 1k samples x 10 = 10k total
        update_batch!(wc2, X_small)
    end
    
    after_small_batches = memory_usage_mb()
    println("After small batches: $(Base.round(after_small_batches, digits=2)) MB")
    println("Memory increase: $(Base.round(after_small_batches - before_small_batches, digits=2)) MB")
    
    # Test 3: Repeated small batches (to detect accumulation)
    println("\n=== Test 3: Repeated small batches (leak detection) ===")
    wc3 = WelfordEstimate(n_features)
    
    memory_readings = Float64[]
    
    for round in 1:5
        GC.gc()
        memory_before_round = memory_usage_mb()
        
        for i in 1:20
            X_batch = randn(batch_size, n_features)
            update_batch!(wc3, X_batch)
        end
        
        GC.gc()
        memory_after_round = memory_usage_mb()
        push!(memory_readings, memory_after_round)
        
        println("Round $round: $(Base.round(memory_after_round, digits=2)) MB")
    end
    
    # Check for memory growth pattern
    if length(memory_readings) >= 2
        memory_growth = memory_readings[end] - memory_readings[1]
        println("\nMemory growth over 5 rounds: $(Base.round(memory_growth, digits=2)) MB")
        
        if memory_growth > 1.0  # More than 1MB growth indicates potential leak
            println("⚠️  POTENTIAL MEMORY LEAK DETECTED")
        else
            println("✅ No significant memory leak detected")
        end
    end
    
    # Test 4: Compare final statistics to ensure correctness
    println("\n=== Test 4: Statistics verification ===")
    stats1 = get_statistics(wc1)
    stats2 = get_statistics(wc2)
    
    mean_diff = maximum(abs.(stats1.mean - stats2.mean))
    cov_diff = maximum(abs.(stats1.covariance - stats2.covariance))
    
    println("Mean difference: $(mean_diff)")
    println("Covariance difference: $(cov_diff)")
    
    if mean_diff < 1e-10 && cov_diff < 1e-10
        println("✅ Statistics are identical between batch methods")
    else
        println("⚠️  Statistics differ between batch methods")
    end
    
    return memory_readings
end

# Run the test
memory_readings = test_memory_leak()