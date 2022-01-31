using StatsFuns
using StatsBase
using Statistics
using Distributed
using Printf
using Crayons
using Base.Iterators
using Dates

# Probably will not be used in favor of the weighted mean/cov functions provided by
# StatsBase
function pmcmc_mean_cov(param_matrix::Array{Float64,2}, log_w::Vector{Float64})
    mean_vec = fill(0.0, size(param_matrix)[1])
    cov_mat = fill(0.0, size(param_matrix)[1], size(param_matrix)[1])
    total_weight = 0.0
    w_max = maximum(log_w)
    w_n = Vector{Float64}(undef, size(param_matrix)[2])
    for n = 1:size(param_matrix)[2]
        w_n[n] = exp(log_w[n] - w_max)
        total_weight += w_n[n]
        mean_vec .+= w_n[n] * (@view param_matrix[:, n])
    end
    mean_vec ./= total_weight
    p_vec = Vector{Float64}(undef, size(param_matrix)[1])
    for n = 1:size(param_matrix)[2]
        p_vec = param_matrix[:, n] .- mean_vec
        cov_mat += w_n[n] * (p_vec * p_vec')
    end
    cov_mat ./= total_weight
    return (mean_vec, cov_mat)
end

function pmcmc_mean_cov(params::Array{Array{Q,1},1}, log_w::Vector{Q}) where {Q<:Any}
    pmcmc_mean_cov(hcat(params...), log_w)
end

function pmcmc_mean_cov(params::Vector{T}, log_w::Vector{Q}) where {T<:NamedTuple,Q<:Any}
    pmcmc_mean_cov(hcat(map(x -> vcat(x...), params)...), log_w)
end

# Given a parameter (for dispatch), sample from a truncated MVT normal distribution
function trunc_normal_jump(
    x::Vector,
    mu::Vector,
    sigma::Matrix;
    lower_bound = (x, n) -> -Inf,
    upper_bound = (x, n) -> Inf,
    perturb = 1e-4,
)
    sigma = sigma + I * perturb
    A = Array{Float64,2}(undef, length(x) - 1, length(x))
    sigma_n = Array{Float64,1}(undef, length(x))
    n_x = length(x)
    x_current = copy(x)
    for n = 1:length(x)
        A = sigma[1:n_x.!=n, 1:n_x.!=n] \ sigma[1:n_x.!=n, n]
        sigma_n = sqrt(sigma[n, n] - (sigma[n, 1:n_x.!=n]' * A))
        mean_n = mu[n] .+ (A'*(x_current[1:n_x.!=n].-mu[1:n_x.!=n]))[1, 1]
        l = lower_bound(x_current, n)
        u = upper_bound(x_current, n)
        x_current[n] = rand(Truncated(Normal(mean_n, sigma_n), l, u))
    end
    return x_current
end


function trunc_normal_dens(
    x::Vector,
    x2::Vector,
    mu::Vector,
    sigma::Matrix;
    lower_bound = (x, n) -> -Inf,
    upper_bound = (x, n) -> Inf,
    perturb = 1e-4,
)
    sigma = sigma + I * perturb
    A = Array{Float64,2}(undef, length(x) - 1, length(x))
    sigma_n = Array{Float64,1}(undef, length(x))
    n_x = length(x)
    x_current = copy(x)
    ldens = 0.0
    for n = 1:length(x)
        A[:, n] = sigma[1:n_x.!=n, 1:n_x.!=n] \ sigma[1:n_x.!=n, n]
        sigma_n[n] = sqrt(sigma[n, n] - (sigma[n, 1:n_x.!=n]' * A[:, n]))
        mean_n = mu[n] .+ (A[:, n]'*(x_current[1:n_x.!=n].-mu[1:n_x.!=n]))[1, 1]
        l = lower_bound(x_current, n)
        u = upper_bound(x_current, n)
        ldens += logpdf(Truncated(Normal(mean_n, sigma_n[n]), l, u), x2[n])
        x_current[n] = x2[n]
    end
    return ldens
end

function smc_pmcmc_proposal(
    theta_current::Vector,
    pmcmc_mean::Vector,
    pmcmc_cov::Array;
    C = (2.38^2) / length(theta_current),
    kwargs...,
)
    return trunc_normal_jump(theta_current, theta_current, pmcmc_cov * C; kwargs...)
end

function smc_pmcmc_proposal_logdens(
    theta_current::Vector,
    theta_proposed::Vector,
    pmcmc_mean::Vector,
    pmcmc_cov::Matrix;
    C = (2.38^2) / length(theta_current),
    kwargs...,
)
    return trunc_normal_dens(
        theta_current,
        theta_proposed,
        theta_current,
        pmcmc_cov * C;
        kwargs...,
    )
end

# Utilities

function calculate_ess(xs)
    x_max = maximum(xs)
    x_sum = 0.0
    x_squared_sum = 0.0
    for n = 1:length(xs)
        exp_x_n = exp(xs[n] - x_max)
        x_sum += exp_x_n
        x_squared_sum += exp_x_n^2
    end

    return x_sum^2 / x_squared_sum
end

# This is a batch function which will run a filter from 1:T for a given vector of parameters in parallel

function dt_smc2_estimation(
    thetas,
    loglik_fun,
    prior_fun;
    grid_steps = 1e-8,
    ess_threshold = 0.5,
    pmcmc_theta_steps = 10,
    parallelize = true,
    kwargs...,
)
    N_θ = length(thetas)
    if parallelize
        ## Currently this pmap is type unstable, but this should
        ## not be too detrimental to performance

        ## Need to make a CachingPool so data is cached
        wp = CachingPool(workers())
        logliks = pmap(wp, thetas) do theta
            loglik_fun(theta)
        end
    else
        wp = missing
        logliks = map(loglik_fun, thetas)
    end
    ξ = 0.0

    acceptances = Array{Bool,3}(undef, N_θ, pmcmc_theta_steps, 0)

    while ξ < 1.0
        print(crayon"blue", "Picking new value for ξ\n")
        ξ_diff = grid_steps
        ess = calculate_ess(logliks * ξ_diff)

        ξ_diff, ess = xi_grid_search(logliks, ess_threshold)

        ξ = min(ξ_diff + ξ, 1.0)

        print(crayon"blue", @sprintf("Xi set to %.3e\nESS = %.3f\n", ξ, ess))

        thetas, logliks, acceptances_i = density_tempered_pmcmc(
            thetas,
            logliks,
            loglik_fun,
            prior_fun,
            ξ,
            ξ_diff;
            wp = wp,
            pmcmc_theta_steps = pmcmc_theta_steps,
            kwargs...,
        )
        acceptances = cat3(acceptances, acceptances_i)
    end

    return thetas, logliks, acceptances
end

function density_tempered_pmcmc(
    thetas,
    logliks,
    loglik_fun,
    prior_fun,
    ξ,
    ξ_diff;
    wp = missing,
    pmcmc_theta_steps = 10,
    rprop_pmh = smc_pmcmc_proposal,
    dprop_pmh = smc_pmcmc_proposal_logdens,
    kwargs...,
)
    N_θ = length(thetas)
    log_ws = logliks * ξ_diff

    ws = exp.(log_ws .- maximum(log_ws))
    pmcmc_mean, pmcmc_cov = pmcmc_mean_cov(thetas, log_ws)
    println("Param" * repeat(" ", 11) * "\tMean\t\tStd\t\t")
    for i = 1:length(pmcmc_mean)
        name = first(string(keys(thetas[1])[i]), 16)
        name *= repeat(" ", 17 - length(name))
        p_str = @sprintf("%s\t%.3e\t%.3e", name, pmcmc_mean[i], sqrt(pmcmc_cov[i, i]))
        println(p_str)
    end

    # A                 = wsample(1:N_θ, ws, N_θ)
    A = residual_resample(Random.GLOBAL_RNG, ws, N_θ)
    thetas = thetas[A]
    logliks = logliks[A]

    acceptances = Array{Bool,2}(undef, N_θ, pmcmc_theta_steps)
    for step_idx = 1:pmcmc_theta_steps
        starttime = Dates.now()
        if !ismissing(wp)
            res = pmap(wp, thetas, logliks) do theta, loglik
                pmcmc_propose_accept_reject(theta, loglik, loglik_fun, prior_fun, ξ, pmcmc_mean, pmcmc_cov, rprop_pmh, dprop_pmh)
            end
        else
            res = map(
                pmcmc_propose_accept_reject,
                thetas,
                logliks,
                cycle([loglik_fun]),
                cycle([prior_fun]),
                cycle([ξ]),
                cycle([pmcmc_mean]),
                cycle([pmcmc_cov]),
                cycle([rprop_pmh]),
                cycle([dprop_pmh]),
            )
        end
        for (i, r) in enumerate(res)
            thetas[i] = r[1]
            logliks[i] = r[2]
            acceptances[i, step_idx] = r[3]
        end
        timetaken = Dates.now() - starttime
        print(
            crayon"green",
            @sprintf(
                "Step %02d: Accepted %0.2f; Filter time: %f secs \n",
                step_idx,
                mean(acceptances[:, step_idx]),
                timetaken.value / 1000
            )
        )
    end
    print(
        crayon"green",
        @sprintf("Total θ moved at least once: %0.2f \n", mean(any(acceptances, dims = 2)))
    )
    return thetas, logliks, acceptances
end

function pmcmc_propose_accept_reject(
    theta,
    loglik,
    loglik_fun,
    prior_fun,
    ξ,
    pmcmc_mean,
    pmcmc_cov,
    rprop_pmh,
    dprop_pmh,
)
    # Propose new parameter
    theta_prop = rprop_pmh(theta, pmcmc_mean, pmcmc_cov)
    d_prop = dprop_pmh(theta, theta_prop, pmcmc_mean, pmcmc_cov)
    d_current = dprop_pmh(theta_prop, theta, pmcmc_mean, pmcmc_cov)

    # Evaluate likelihood of parameter
    loglik_prop = loglik_fun(theta_prop)

    # Evaluate prior of parameter
    prior = prior_fun(theta)
    prior_prop = prior_fun(theta_prop)

    # Calculate log acceptance probability
    log_prior_ratio = prior_prop - prior
    log_lik_ratio = ξ * (loglik_prop - loglik)
    log_prop_ratio = d_current - d_prop

    log_posterior_proposed = loglik_prop + prior_prop
    log_acceptance = log_prior_ratio + log_lik_ratio + log_prop_ratio
    accepted = (log_posterior_proposed > -Inf && log(rand()) < log_acceptance)

    if accepted
        theta = theta_prop
        loglik = loglik_prop
    end

    return (theta, loglik, accepted)
end

function xi_grid_search(w::AbstractArray{Float64}, ess_threshold = 0.5, depth = 100)
    ξ = (2.0)^(-1.0)
    for d = 2:depth
        ess = calculate_ess(w * ξ) / length(w)
        if ess < ess_threshold
            ξ -= (2.0)^(-d)
        else
            ξ += (2.0)^(-d)
        end
    end

    ess = calculate_ess(w * ξ) / length(w)
    if (ess < ess_threshold)
        ξ -= (2.0)^(-depth)
    end

    if ξ == 0
        println("Less than ess_threshold of observations are zero")
        ξ = 1e-12
    end

    return ξ, ess
end
