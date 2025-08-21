module RunningStats

    using LinearAlgebra
    using Statistics

    include("welford_covariance.jl")

    export WelfordEstimate,
        initialize!,
        update_batch!,
        update_single!,
        get_covariance,
        get_correlation,
        get_statistics,
        merge_estimate!,
        merge_estimate

end