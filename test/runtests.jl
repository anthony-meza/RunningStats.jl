using Test
using RunningStats
using Random
using LinearAlgebra
using Statistics

@testset "RunningStats.jl Tests" begin
    include("test_basic_functionality.jl")
    include("test_welford_covariance.jl")

end