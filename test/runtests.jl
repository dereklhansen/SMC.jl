using SMC
using Test

@testset "kalman" begin
    include("kalman.jl")
end
@testset "smc" begin
    include("smc.jl")
end
