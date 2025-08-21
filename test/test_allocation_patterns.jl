using Statistics, LinearAlgebra, Random
using Distributions
include("../src/welford_covariance.jl")

# Helper function to check if a number is a natural number
is_natural_number(x) = isa(x, Integer) && x > 0

# Your WelfordCovariance function
function WelfordCovariance(sampling_function::Function, 
                            sampling_distribution;  
                          batch_size = nothing, 
                          total_samples = nothing)
    if !is_natural_number(batch_size)
        throw(ArgumentError("batch_size must be a positive integer"))
    end
    if !is_natural_number(total_samples)
        throw(ArgumentError("total_samples must be a positive integer"))
    end
    if total_samples % batch_size != 0
        throw(ArgumentError("batch_size ($batch_size) must divide total_samples ($total_samples) evenly"))
    end

    if !isa(sampling_distribution, Distribution)
        throw(ArgumentError("sampling_distribution must be a Distribution"))
    end

    feature_length = length(sampling_function(rand(sampling_distribution)))
    ensemble = Matrix{Float64}(undef, batch_size, feature_length)
    nstrides = div(total_samples, batch_size)
    random_samples = Vector{Float64}(undef, batch_size)
    estimator = WelfordEstimate()

    for k in 1:nstrides
        rand!(sampling_distribution, random_samples)
        for i in eachindex(random_samples)
            @views ensemble[i, :] = sampling_function(random_samples[i])
        end
        update_batch!(estimator, ensemble)
    end
    return get_covariance(estimator)
end

function test_allocation_patterns()
    println("Testing allocation patterns with @time...")
    
    # Test function: simple polynomial features
    poly_features(x) = [x, x^2, x^3, x^4]
    dist = Normal(0, 1)
    
    println("\n=== Testing consistent allocations across repeated calls ===")
    
    # Test with same parameters multiple times
    batch_size = 100
    total_samples = 1000
    
    allocations = Int[]
    
    for i in 1:5
        println("\nRun $i:")
        stats = @time WelfordCovariance(poly_features, dist; 
                                      batch_size=batch_size, 
                                      total_samples=total_samples)
        
        # Extract allocation info from @time output
        # Note: We'll collect this manually since @time prints to stdout
        # For automated testing, we'd use @allocated macro instead
    end
    
    println("\n=== Using @allocated for precise measurement ===")
    
    # Use @allocated to get exact allocation numbers
    for i in 1:5
        alloc = @allocated WelfordCovariance(poly_features, dist; 
                                          batch_size=batch_size, 
                                          total_samples=total_samples)
        push!(allocations, alloc)
        println("Run $i: $(alloc) bytes allocated")
    end
    
    # Check consistency
    if length(unique(allocations)) == 1
        println("✅ Allocations are consistent: $(allocations[1]) bytes")
    else
        println("⚠️  Allocation inconsistency detected:")
        println("   Min: $(minimum(allocations)) bytes")
        println("   Max: $(maximum(allocations)) bytes") 
        println("   Std: $(std(allocations)) bytes")
    end
    
    println("\n=== Testing allocation scaling with batch size ===")
    
    batch_sizes = [50, 100, 200, 400]
    scaling_allocations = Int[]
    
    for bs in batch_sizes
        ts = bs * 10  # Keep same number of strides
        alloc = @allocated WelfordCovariance(poly_features, dist; 
                                          batch_size=bs, 
                                          total_samples=ts)
        push!(scaling_allocations, alloc)
        println("Batch size $bs: $(alloc) bytes")
    end
    
    # Check if allocations scale reasonably with batch size
    println("\nAllocation scaling analysis:")
    for i in 2:length(batch_sizes)
        ratio = scaling_allocations[i] / scaling_allocations[1]
        size_ratio = batch_sizes[i] / batch_sizes[1]
        println("  $(batch_sizes[i])x batch size: $(round(ratio, digits=2))x allocations (expected ≈$(size_ratio)x)")
    end
    
    println("\n=== Testing allocation scaling with total samples ===")
    
    total_sample_counts = [1000, 2000, 4000, 8000]
    sample_allocations = Int[]
    
    for ts in total_sample_counts
        alloc = @allocated WelfordCovariance(poly_features, dist; 
                                          batch_size=100, 
                                          total_samples=ts)
        push!(sample_allocations, alloc)
        println("Total samples $ts: $(alloc) bytes")
    end
    
    println("\nSample count scaling analysis:")
    for i in 2:length(total_sample_counts)
        ratio = sample_allocations[i] / sample_allocations[1]
        sample_ratio = total_sample_counts[i] / total_sample_counts[1]
        println("  $(sample_ratio)x samples: $(round(ratio, digits=2))x allocations")
        
        if ratio > sample_ratio * 1.1  # Allow 10% tolerance
            println("    ⚠️  Higher than expected allocation growth!")
        else
            println("    ✅ Reasonable allocation scaling")
        end
    end
    
    println("\n=== Testing for memory leaks in repeated calls ===")
    
    # Test repeated calls to see if allocations accumulate
    println("Running 10 identical calls to detect accumulation...")
    leak_allocations = Int[]
    
    for i in 1:10
        alloc = @allocated WelfordCovariance(poly_features, dist; 
                                          batch_size=100, 
                                          total_samples=1000)
        push!(leak_allocations, alloc)
        if i % 3 == 0
            println("  Calls 1-$i: $(leak_allocations[1]) to $(alloc) bytes")
        end
    end
    
    if maximum(leak_allocations) - minimum(leak_allocations) < 1000  # Less than 1KB variation
        println("✅ No memory leak detected - consistent allocations")
    else
        println("⚠️  Potential memory leak - allocation variation: $(maximum(leak_allocations) - minimum(leak_allocations)) bytes")
    end
    
    return (allocations, scaling_allocations, sample_allocations, leak_allocations)
end

# Run the test
Random.seed!(12345)
test_allocation_patterns()