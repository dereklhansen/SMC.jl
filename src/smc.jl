# Named Tuple Macro
macro NT(expr...)
    nmexpr = Expr(:curly, esc(NamedTuple), :($expr))
    tpexpr = Expr(:tuple, (esc(e) for e in expr)...)
    Expr(:call, nmexpr, tpexpr)
end

cpu = identity



exp_mean(xs) = logsumexp(xs) - log(length(xs))
ess_exp(xs) = exp(2 * logsumexp(xs) - logsumexp(2 * xs))

pass_threshold(logweights, threshold) =
    ess_exp(logweights) >= threshold * length(logweights)

function adaptive_resample_particles(
    rng,
    logweights,
    ancestors,
    atable,
    old_particles,
    threshold = 1.0,
)
    if pass_threshold(logweights, threshold)
        return old_particles, logweights, false
    else
        resampled_particles =
            resample_particles(rng, cpu(logweights), ancestors, atable, old_particles)
        return resampled_particles, zero(logweights), true
    end
end

dpre_default(Xs_prev, t) = zeros(eltype(Xs_prev), size(Xs_prev, 2))

Zygote.@nograd pass_threshold

function setindex_ng!(xs, i, x)
    xs[i] = x
    return nothing
end

Zygote.@nograd setindex_ng!

function getindex_ng(xs, is)
    return xs[is]
end

Zygote.@nograd getindex_ng


function smc(
    rng,
    end_t,
    rinit,
    rproposal,
    dinit,
    dprior,
    dproposal,
    dtransition,
    dmeasure,
    dpre = dpre_default;
    record_history = false,
    threshold = 1.0,
)
    particles = rinit(rng)
    D, n_particles = size(particles)
    logK = convert(typeof(particles[1]), log(n_particles))
    # Calculate log-likelihood of first particles
    logweights = dmeasure(particles, 1) + dprior(particles) - dinit(particles)
    ancestors = zeros(Int, n_particles)

    if (record_history)
        particle_history = repeat(particles, outer = [1, 1, end_t])
        logweight_history = repeat(logweights, outer = [1, end_t])
        ancestor_history = zeros(Int, n_particles, end_t)
    else
        particle_history = missing
        logweight_history = missing
        ancestor_history = missing
    end

    atable = AliasTable(logweights[1], n_particles)

    loglik = exp_mean(logweights)
    ess = zeros(typeof(logweights[1]), end_t)
    logliks = zeros(typeof(logweights[1]), end_t)
    fq_ratio = zeros(typeof(logweights[1]), end_t)
    resampled = fill(false, end_t)

    setindex_ng!(logliks, 1, loglik)
    setindex_ng!(fq_ratio, 1, zero(fq_ratio[1]))
    setindex_ng!(ess, 1, ess_exp(logweights))

    particles, logweights = let old_particles = particles, logweights = logweights
        for t = 2:end_t
            ## Resampling
            logweights = logweights .- logsumexp(logweights)
            g_pre = dpre(old_particles, t)
            @assert !any(isnan.(g_pre))
            logweights += g_pre
            ll_pre = logsumexp(logweights)


            resampled_particles, logweights_prev, resampled_t =
                adaptive_resample_particles(
                    rng,
                    logweights,
                    ancestors,
                    atable,
                    old_particles,
                    threshold,
                )
            setindex_ng!(resampled, t, resampled_t)
            if resampled_t
                g_pre = getindex_ng(g_pre, ancestors)
            end
            new_particles = rproposal(rng, resampled_particles, t)
            f_trans = dtransition(resampled_particles, new_particles, t)
            @assert !any(isnan.(f_trans))
            g_measure = dmeasure(new_particles, t)
            @assert !any(isnan.(g_measure))
            q_prop = dproposal(resampled_particles, new_particles, t)
            @assert !any(isnan.(q_prop))
            f_over_q = f_trans - q_prop
            logweights = f_trans + g_measure - q_prop - g_pre + logweights_prev

            logliks_t = logsumexp(logweights) - logsumexp(logweights_prev) + ll_pre
            loglik += logliks_t

            if loglik == -Inf
                break
            end

            setindex_ng!(logliks, t, logliks_t)
            setindex_ng!(fq_ratio, t, exp_mean(f_over_q))
            setindex_ng!(ess, t, ess_exp(logweights))

            if record_history
                particle_history[:, :, t] .= new_particles
                logweight_history[:, t] .= logweights
                ancestor_history[:, t] = ancestors
            end

            old_particles = new_particles
        end
        particles, logweights
    end
    return @NT(
        loglik,
        logliks,
        particles,
        logweights,
        ess,
        fq_ratio,
        resampled,
        particle_history,
        logweight_history,
        ancestor_history
    )
end

## Stripped down version which only returns the log-likelihood and more
## amenable to autodiff

function resample_particles(rng, logweights, ancestors, atable, old_particles)
    n_particles = length(logweights)
    lw_max = maximum(logweights)
    W = exp.(logweights .- lw_max)
    residual_resample!(rng, 1:n_particles, W, ancestors, atable)

    resampled_particles = (old_particles[:, ancestors])
    return resampled_particles
end

Zygote.@nograd resample_particles

function calc_filtered_mean(particle_history, logweight_history)
    D, K, T = size(particle_history)
    mu = fill(particle_history[1], D, T)
    for t = 1:T
        w = exp.(@view(logweight_history[:, t]) .- maximum(@view(logweight_history[:, t])))
        mu[:, t] = sum(particle_history[:, i, t] * w[i] for i = 1:K) / sum(w)
    end
    return mu
end

function simulate_backward(rng, particle_history, logweight_history, dtransition, n_draws)
    D, n_particles, T = size(particle_history)
    X = particle_history
    LW = logweight_history

    w = exp.(view(LW, :, T) .- maximum(view(LW, :, T)))
    a = residual_resample!(rng, 1:n_particles, w, Vector{Int}(undef, n_draws))

    X_sm = X[:, a, :]

    F = fill(zero(X[1]), n_draws, n_particles)

    for t = (T-1):-1:1
        F = dtransition_outer(view(X, :, :, t), view(X_sm, :, :, t + 1), dtransition, t + 1)
        LW_sm = calculate_ffbs_weights(view(LW, :, t), F)
        for j = 1:n_draws
            lw = view(LW_sm, :, j)
            w = exp.(lw .- maximum(lw))
            a = wsample(rng, 1:n_particles, w)
            X_sm[:, j, t] = X[:, a, t]
        end
    end

    return X_sm
end

function dtransition_outer(Xprev, X, dtransition, t)
    n_particles = size(Xprev)[2]
    n_draws = size(X)[2]
    X1 = view(Xprev, :, repeat(1:n_particles, outer = n_draws))
    X2 = view(X, :, repeat(1:n_draws, inner = n_particles))
    F = dtransition(X1, X2, t)
    return reshape(F, n_particles, n_draws)
end

function calculate_ffbs_weights(
    ws_x::AbstractVector,
    ws_y::AbstractVector,
    F::AbstractMatrix,
)
    ws_x .+ F .+ ws_y'
end

function calculate_ffbs_weights(ws_x::AbstractVector, F::AbstractMatrix)
    ws_x .+ F
end

function simulate_threaded(particle_history, logweight_history, dt, K)
    N = Base.Threads.nthreads()
    ffbs_out = zeros(
        eltype(particle_history),
        size(particle_history, 1),
        K * N,
        size(particle_history, 3),
    )
    Base.Threads.@threads for n = 1:N
        rng = Random.MersenneTwister()
        ffbs_out[:, (1+(n-1)*K):(n*K), :] =
            simulate_backward(rng, particle_history, logweight_history, dt, K)
    end
    return ffbs_out
end
