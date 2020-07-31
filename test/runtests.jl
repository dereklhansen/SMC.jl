using SMC
using Test

@testset "SMC.jl" begin
    @testset "kalman" begin
        include("kalman.jl")
    end
end
