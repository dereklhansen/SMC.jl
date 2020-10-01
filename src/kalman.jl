# Kalman filter code.
# Please see Lopes Tsay 2011 "Particle Filters and Bayesian Inference In Financial Econometrics"
using LinearAlgebra

function push_state_forward(G0, G1, Tau, m, V)
    a_t = G0 + G1 * m
    R_t = G1 * Symmetric(V) * G1' + Symmetric(Tau)
    return a_t, R_t
end

function calc_loglik_t(F0, F1, Σ, a_t, R_t, y_t)
    f_t = F0 + F1 * a_t
    Q_t = F1 * Symmetric(R_t) * F1' + Symmetric(Σ)
    e_t = (y_t - f_t)
    loglik = logpdf(MvNormal(Symmetric(Q_t)), e_t)[1]
    return e_t, Q_t, loglik
end

function update_state(F1, a_t, R_t, e_t, Q_t)
    # Kalman Gain
    A_t = R_t * F1' / Q_t
    m_t = a_t + A_t * e_t
    V_t = Symmetric(R_t) - A_t * Symmetric(Q_t) * A_t'
    return m_t, V_t
end




function iterate_kalman_filter_mv(F0, F1, G0, G1, Σ, Tau, m, V, y_t)
    # Push the state forward
    a_t, R_t = push_state_forward(G0, G1, Tau, m, V)
    # Calculate the likelihood
    e_t, Q_t, loglik = calc_loglik_t(F0, F1, Σ, a_t, R_t, y_t)
    # Calculate the Kalman Gain
    m_t, V_t = update_state(F1, a_t, R_t, e_t, Q_t)

    return m_t, V_t, loglik
end

function kalman_filter_mv(F0, F1, G0, G1, Σ, Tau, μ0, Tau0, Y)
    T      = size(Y)[1]
    m      = deepcopy(μ0)
    ms     = repeat(m, outer=[1, T])
    V      = deepcopy(Tau0)
    Vs     = repeat(V, outer = [1, 1, T])
    P      = inv(V)
    loglik = zero(μ0[1])

    loglik_t = fill(loglik, T)

    if !(any(ismissing.(Y[1, :])))
        e_t, Q_t, loglik_t[1] = calc_loglik_t(F0(1), F1(1), Σ(1), m, V, @view(Y[1, :]))
        loglik            += loglik_t[1]
        m, V               = update_state(F1(1), m, V, e_t, Q_t)
        ms[:, 1]           = m
        Vs[:, :, 1]         = V
    else
        loglik += 0.0
    end

    if (T > 1)
        for t in 2:T
            # Push system forward
            if !any(ismissing.(Y[t, :]))
                m, V, loglik_t[t] = iterate_kalman_filter_mv(F0(t), F1(t),
                                                          G0(t), G1(t),
                                                          Σ(t), Tau(t),
                                                          m, V, @view(Y[t, :]))
                loglik += loglik_t[t]
            else
                m, V = push_state_forward(G0(t), G1(t), Tau(t), m, V)
            end
            ms[:, t] = m
            Vs[:, :, t] = V
        end
    end

    return loglik, ms, Vs, loglik_t
end

Zygote.@nograd kalman_filter_mv

function kalman_loglik(F0, F1, G0, G1, Σ, Tau, μ0, Tau0, Y)
    T      = size(Y)[1]
    m      = deepcopy(μ0)
    V      = deepcopy(Tau0)
    P      = inv(V)
    loglik = zero(μ0[1])

    if !(any(ismissing.(Y[1, :])))
        e_t, Q_t, loglik_1 = calc_loglik_t(F0(1), F1(1), Σ(1), m, V, @view(Y[1, :]))
        loglik            += loglik_1
        m, V               = update_state(F1(1), m, V, e_t, Q_t)
        # ms[:, 1]           = m
        # Vs[:, :, 1]         = V
    else
        loglik += 0.0
    end

    if (T > 1)
        for t in 2:T
            # Push system forward
            if !any(ismissing.(Y[t, :]))
                m, V, loglik_t = iterate_kalman_filter_mv(F0(t), F1(t),
                                                          G0(t), G1(t),
                                                          Σ(t), Tau(t),
                                                          m, V, @view(Y[t, :]))
                loglik += loglik_t
            else
                m, V = push_state_forward(G0(t), G1(t), Tau(t), m, V)
            end
            # ms[:, t] = m
            # Vs[:, :, t] = V
        end
    end

    return loglik
end
## Smoother

function kalman_smoother_mv(F0, F1, G0, G1, Σ, Tau, μ0, Tau0, Y)
    T              = size(Y)[1]

    loglik, ms, Vs = kalman_filter_mv(F0, F1, G0, G1, Σ, Tau, μ0, Tau0, Y)
    D              = size(ms, 1)
    ms_smoothed    = deepcopy(ms)
    Vs_smoothed    = deepcopy(Vs)

    ms_smoothed[:, 1:(end-1)] .= NaN
    Vs_smoothed[:, :, 1:(end-1)] .= NaN

    Covs_smoothed = zeros(eltype(ms_smoothed), D, D, T-1)

    if (T > 1)
        for t in (T-1):-1:1
            m_smoothed, V_smoothed, Cov_smoothed = backward_smooth_step_mv(G0(t+1), G1(t+1),
                                                             Tau(t+1),
                                                             view(ms, :, t),
                                                             view(Vs, :, :, t),
                                                             view(ms_smoothed, :, t+1),
                                                             view(Vs_smoothed, :, :, t+1))
            ms_smoothed[:, t] = m_smoothed
            Vs_smoothed[:, :, t] = V_smoothed
            Covs_smoothed[:, :, t] = Cov_smoothed
        end
    end

    return ms_smoothed, Vs_smoothed, Covs_smoothed
end

Zygote.@nograd kalman_smoother_mv


function backward_smooth_step_mv(G0, G1, Tau, m_t, V_t, m_tp1_sm, V_tp1_sm)
    # One step ahead predictive
    a_tp1, R_tp1 = push_state_forward(G0, G1, Tau, m_t, V_t)
    B              = (V_t * G1') / R_tp1
    m_t_sm         = m_t + B * (m_tp1_sm - a_tp1)
    V_t_sm         = V_t + B * (V_tp1_sm - R_tp1) * B'
    Cov_tp1_t      = V_tp1_sm * B'

    return m_t_sm, V_t_sm, Cov_tp1_t
end

function draw_posterior_path(rng, ms, Vs, G0, G1, Σ, Tau, μ0, Tau0)
    T              = size(ms)[2]

    Xs_smoothed    = deepcopy(ms)
    V_fill_in      = zero(Vs[:, :, end])

    Xs_smoothed[:, end] = rand(rng, MvNormal(ms[:, end], Symmetric(Vs[:, :, end])))

    if (T > 1)
        for t in (T-1):-1:1
            m_smoothed, V_smoothed, _ = backward_smooth_step_mv(G0(t+1), G1(t+1),
                                                             Tau(t+1),
                                                             view(ms, :, t),
                                                             view(Vs, :, :, t),
                                                             view(Xs_smoothed, :, t+1),
                                                             V_fill_in)
            Xs_smoothed[:, t] = rand(rng, MvNormal(m_smoothed, Symmetric(V_smoothed)))
        end
    end

    return Xs_smoothed
end
