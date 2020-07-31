module SMC

using Random
using Zygote
using StatsFuns
using Statistics
using StatsBase
using Distributions
using LinearAlgebra
using Printf
using Distributed
using Crayons
using Base.Iterators
using Dates

include("smc.jl")
include("kalman.jl")
include("smc2_inference.jl")

end
