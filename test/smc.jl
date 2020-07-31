#= using Pkg =#
#= Pkg.activate(".") =#
#= Pkg.instantiate() =#
using Pkg
using RCall
using JSON
using Memoization
using Test
using DelimitedFiles

include("../src/kalman.jl")
include("../src/make_kalman_model.jl")
include("../src/kalman_train.jl")
include("../src/PV_source.jl")
include("../src/smc.jl")
include("../src/models.jl")

kalman_dir = pwd()
root_dir   = kalman_dir * "/../.."

R"
setwd($root_dir)
source('code/src/import_data_box.R')
prof_data  <- import_data_box('profile_data.RDS', readRDS, use_boxr = FALSE)
float_data <- import_data_box('float_data.RDS', readRDS, use_boxr = FALSE)
NULL
"

cd(kalman_dir)

@rget prof_data
@rget float_data

## Floats
example_float_id = "5901717"
prof             = prof_data[prof_data.float .== example_float_id, :]
myfloat            = float_data[float_data.float .== parse(Float64, example_float_id), :]

X            = Matrix{Union{Missing, Float64}}(myfloat[:, [:long, :lat]])
interp       = Vector(myfloat[:, :pos_qc]) .== "8"
X[interp, :]   .= missing

X

days   = Vector(myfloat.day)
F0, F, G0, G, Tau, Σ = make_kalman_model(days, 3e-3, 3e-3, 3e-3, 3e-3, 1e-4)

# Make the initial state and variance just the first observation for now
μ0   = convert(Vector{Float64}, vcat(X[1, :], [0.0, 0.0]))
Tau0 = diagm([1e-4, 1e-4, 1e-4, 1e-4])

# Set first obs to missing to make up for this
X_in     = deepcopy(X)
X_in[1,:].=missing

## MLE
using Flux

res = train_argo_kalman(X, 3e-3, 3e-3, 3e-3, 3e-3, 1e-4, μ0, Tau0, days, 1000, Flux.Descent(1e-2), 10)

σ_x_long_hat, σ_x_lat_hat, σ_v_long_hat, σ_v_lat_hat, σ_p_hat = exp.(res)

F0, F, G0, G, Tau, Σ = make_kalman_model(days, σ_x_long_hat, σ_x_lat_hat, σ_v_long_hat, σ_v_lat_hat, σ_p_hat)
ll, states, variances, lls = kalman_filter_mv(F0, F, G0, G, Σ, Tau, μ0, Tau0, X_in)
states_sm, variances_sm, Covs_sm = kalman_smoother_mv(F0, F, G0, G, Σ, Tau, μ0, Tau0, X_in)

## Now we use the smoothed kalman filter within SMC
K         = 20
T         = size(X_in, 1)

G_X    = t -> getindex(G(t), 1:2, :)
Tau_X  = t -> Symmetric(getindex(Tau(t), 1:2, 1:2))
Tau_PV = t -> Symmetric(getindex(Tau(t), 3:4, 3:4))

γ         = 1.0
rng       = MersenneTwister(342)

θ = (σ_x_long = σ_x_long_hat,
     σ_x_lat = σ_x_lat_hat,
     σ_v_long = σ_v_long_hat,
     σ_v_lat = σ_v_lat_hat,
     σ_p = σ_p_hat,
     γ = γ)


b       = 300
PV_grad = readdlm(string("../../temp/data/PV", b, ".csv"), ',')
PV_grad = DataFrame(long = PV_grad[:,1],
                    lat = PV_grad[:, 2],
                    long_grad = PV_grad[:,3],
                    lat_grad = PV_grad[:,4])

modeltype   = ArgoModels.ArgoBaseline()
smc_model = ArgoModels.smc_model(modeltype, θ, (X_in = deepcopy(X), days=days, PV_grad=PV_grad), K)
Xs      = smc_model.rinit(rng)
dprs    = smc_model.dpr(Xs)
dm1     = smc_model.dm(Xs, 1)
dinits  = smc_model.dinit(Xs)
dpres   = smc_model.dpre(Xs, 2)

@test size(Xs) == (6, K)

dpr_actual = logpdf(MvNormal(μ0, Tau0), Xs[1:4, :])
@test dprs ≈ dpr_actual

dm_actual = fill(0.0, K)
@test dm1 ≈ dm_actual

## The adapted filter should have all weights equal to the total-data likelihood
logweights = dprs + dm1 - dinits + dpres
@test all(logweights .≈ ll)

Xs_new = smc_model.rp(rng, Xs, 2)
dts    = smc_model.dt(Xs, Xs_new, 2)
dm2    = smc_model.dm(Xs_new, 2)
dps    = smc_model.dp(Xs, Xs_new, 2)

using Debugger

@enter smc_model.rp(rng, Xs, 2)

@enter smc_model.dt(Xs, Xs_new, 2)

@test size(Xs_new) == (6, K)

dt_actual  = logpdf(MvNormal(Tau(2)), Xs_new[1:4, :] - G(2)*Xs[1:4, :])
@test dts ≈ dt_actual

dm_actual = fill(0.0, K)
@test dm2 ≈ dm_actual

mu_proposal  = states_sm[:, 2] .+ (Covs_sm[:, :, 1] / variances_sm[:, :, 1]) * (Xs[1:4, :] .- states_sm[1:4, 1])
sig_proposal = variances_sm[:, :, 2] - (Covs_sm[:, :, 1] / variances_sm[:, :, 1]) * Covs_sm[:, :, 1]'

@time smc_out  = smc(rng, T, smc_model...; threshold=0.5, record_history=true);

## There will still be some slight noise in the particle filter, so this approx may fail
@test smc_out.loglik ≈ ll

@time smc_out  = smc(rng, T, smc_model...; record_history=true, threshold=1.0);

## There will still be some slight noise in the particle filter, so this approx may fail
@test smc_out.loglik ≈ ll

fwd_weights = smc_out.logweight_history[:, 27] + smc_model.dpre(smc_out.particle_history[:, :, 27], 28)
fwd_w = fwd_weights .- logsumexp(fwd_weights)
@test all(isapprox.(fwd_w, -log(K)))


## Profiling code
using ProfileView

function profile_smc(rng, T, smc_model, times)
     for t in 1:times
          smc_out = smc(rng, T, smc_model...; record_history=true, threshold=1.0)
     end
end

ProfileView.@profview profile_smc(rng, T, smc_model, 50)
