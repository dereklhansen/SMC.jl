using LinearAlgebra: Symmetric, isposdef
using Distributions: MvNormal, logpdf

import ..kalman_filter_mv
import ..kalman_smoother_mv

abstract type AbstractLinearGaussian end;
struct LinearGaussian <: AbstractLinearGaussian end;


# Generative Model
function dpr_base(::AbstractLinearGaussian, Zs, μ0, Tau0)
    logpdf(MvNormal(μ0, Tau0), Zs)
end

function rpr_base(::AbstractLinearGaussian, rng, K, states, variances)
    rand(rng, MvNormal(states[:, 1], Symmetric(variances[:, :, 1])), K)
end

function dt_base(::AbstractLinearGaussian, Zs, Zs_new, t, G, Tau)
    μ      = G(t) * Zs
    Σ      = Tau(t)
    d      = logpdf(MvNormal(Σ), Zs_new - μ)
    @assert !any(isnan.(d))
    return d
end

function dm_base(::AbstractLinearGaussian, Zs, t, F, Σ, Y)
    yt = view(Y, t, :)
    if any(ismissing.(yt))
        return zeros(eltype(Zs), size(Zs, 2))
    else
        d = logpdf(MvNormal(Σ(t)), F(t) * Zs .- yt)
        @assert !any(isnan.(d))
        return d
    end
end

# Inference Model
function rinit_base(::AbstractLinearGaussian, rng, K, states_sm, variances_sm)
    rand(rng, MvNormal(states_sm[:, 1], Symmetric(variances_sm[:, :, 1])), K)
end

function dinit_base(::AbstractLinearGaussian, Zs, states_sm, variances_sm)
    logpdf(MvNormal(states_sm[:, 1], Symmetric(variances_sm[:, :, 1])), Zs)
end

function calc_prop(::AbstractLinearGaussian, Zs, t, states_sm, variances_sm, Covs_sm)
    μ = states_sm[:, t] .+ (Covs_sm[:, :, t-1] / variances_sm[:, :, t-1]) * (Zs .- states_sm[:, t-1])
    Σ = Symmetric(variances_sm[:, :, t] - (Covs_sm[:, :, t-1] / variances_sm[:, :, t-1]) * Covs_sm[:, :, t-1]')

    @assert !any(isnan.(μ))
    @assert !any(isnan.(Σ))

    return μ, Σ
end

function rp_base(m::AbstractLinearGaussian, rng, Zs, t, states_sm, variances_sm, Covs_sm)
    μ, Σ = calc_prop(m, Zs, t, states_sm, variances_sm, Covs_sm)
    @assert isposdef(Σ)
    Zs_new          = μ + rand(rng, MvNormal(Σ), size(Zs, 2))
    return Zs_new
end

function dp_base(m::AbstractLinearGaussian, Zs, Zs_new, t, states_sm, variances_sm, Covs_sm)
    μ, Σ = calc_prop(m, Zs, t, states_sm, variances_sm, Covs_sm)
    @assert isposdef(Σ)
    d = logpdf(MvNormal(Σ), Zs_new - μ)
    return d
end

function dpre_base(::AbstractLinearGaussian, Zs, t, states, variances, states_sm, variances_sm, lls)
    d_smooth = logpdf(MvNormal(states_sm[:, t-1], Symmetric(variances_sm[:, :, t-1])), Zs)
    d_filt   = logpdf(MvNormal(states[:, t-1], Symmetric(variances[:, :, t-1])), Zs)

    d = (d_smooth - d_filt) .+ sum(@view(lls[t:end]))
    @assert !any(isnan.(d))
    return d
end

function make_kalman_proposal(::AbstractLinearGaussian, F0, F, G0, G, Tau, Σ, μ0, Tau0, Y)
    ll, states, variances, lls       = kalman_filter_mv(F0, F, G0, G, Σ, Tau, μ0, Tau0, Y)
    states_sm, variances_sm, Covs_sm = kalman_smoother_mv(F0, F, G0, G, Σ, Tau, μ0, Tau0, Y)
    return @NT(ll, states, variances, lls, states_sm, variances_sm, Covs_sm)
end

function smc_model(m::AbstractLinearGaussian, K, F0, F, G0, G, Tau, Σ, μ0, Tau0, Y)
    ll, states, variances, lls, states_sm, variances_sm, Covs_sm =
        make_kalman_proposal(m, F0, F, G0, G, Tau, Σ, μ0, Tau0, Y)

    dpr(Zs)           = dpr_base(m, Zs, μ0, Tau0)
    dt(Zs, Zs_new, t) = dt_base(m, Zs, Zs_new, t, G, Tau)
    dm(Zs, t)         = dm_base(m, Zs, t, F, Σ, Y)

    rinit(rng)        = rinit_base(m, rng, K, states_sm, variances_sm)
    dinit(Zs)         = dinit_base(m, Zs, states_sm, variances_sm)
    rp(rng, Zs, t)    = rp_base(m, rng, Zs, t, states_sm, variances_sm, Covs_sm)
    dp(Zs, Zs_new, t) = dp_base(m, Zs, Zs_new, t, states_sm, variances_sm, Covs_sm)

    dpre(Zs, t)       = dpre_base(m, Zs, t, states, variances, states_sm, variances_sm, lls)

    return @NT(rinit, rp, dinit, dpr, dp, dt, dm, dpre)
end
