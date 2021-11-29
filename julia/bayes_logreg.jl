# Packages
using LinearAlgebra

# Sigmoid function:
function 𝛔(a)
    trunc = 8.0 # truncation to avoid numerical over/underflow
    a = clamp.(a,-trunc,trunc)
    p = exp.(a)
    p = p ./ (1 .+ p)
    return p
end

# Negative log likelihood
function 𝓁(w,w_0,H_0,X,y)
    N = length(y)
    D = size(X)[2]
    μ = 𝛔(X*w)
    Δw = w-w_0
    return - ∑( y[n] * log(μ[n]) + (1-y[n]) * log(1-μ[n]) for n=1:N) + 1/2 * Δw'H_0*Δw
end

# Negative log likelihood (unconstrained)
function 𝓁_(w,w_0,H_0,X,y)
    N = length(y)
    D = size(X)[2]
    #a = clamp.(X*w, -8.0, 8.0)
    a = X*w
    Δw = w-w_0
    return a'y .- log.(1 .+ exp.(a')) * ones(N) .+ 1/2 * Δw'H_0*Δw
end

# Gradient:
function ∇𝓁(w,w_0,H_0,X,y)
    N = length(y)
    μ = 𝛔(X*w)
    Δw = w-w_0
    g = ∑((μ[n]-y[n]) * X[n,:] for n=1:N)
    return g + H_0*Δw
end

# Hessian:
function ∇∇𝓁(w,w_0,H_0,X,y)
    N = length(y)
    μ = 𝛔(X*w)
    H = ∑(μ[n] * (1-μ[n]) * X[n,:] * X[n,:]' for n=1:N)
    return H + H_0
end

# Main function:
struct BayesLogreg
    μ::Vector{Float64}
    Σ::Matrix{Float64}
end
function bayes_logreg(X,y,w_0,H_0,𝓁,∇𝓁,∇∇𝓁,optim_options...)
    # Model:
    w_map, H_map = newton(𝓁, w_0, ∇𝓁, ∇∇𝓁, (w_0=w_0, H_0=H_0, X=X, y=y), optim_options...) # fit the model (find mode of posterior distribution)
    Σ_map = inv(H_map) # inverse Hessian at the mode
    Σ_map = Symmetric(Σ_map) # to ensure matrix is Hermitian (i.e. avoid rounding issues)
    # Output:
    mod = BayesLogreg(w_map, Σ_map)
    return mod
end

# Methods:
μ(mod::BayesLogreg) = mod.μ
Σ(mod::BayesLogreg) = mod.Σ
# Predict from classifier:
function predict(mod::BayesLogreg, X, proba=true)
    μ = mod.μ # MAP mean vector
    y_hat = 𝛔(X*w)
    if (proba)
        y_hat = round.(y_hat)
    end
    return(y_hat)
end
# Sampling from posterior distribution:
using Distributions
function sample_posterior(mod::BayesLogreg, n)
    rand(MvNormal(mod.μ, mod.Σ),n)
end
# Posterior predictions:
function posterior_predictive(mod::BayesLogreg, X)
    μ = mod.μ # MAP mean vector
    Σ = mod.Σ # MAP covariance matrix
    if !isa(X, Matrix)
        X = reshape(X, 1, length(X))
    end
    # Inner product:
    z = X*μ
    # Probit approximation
    v = [X[n,:]'Σ*X[n,:] for n=1:size(X)[1]]
    κ = 1 ./ sqrt.(1 .+ π/8 .* v) 
    z = κ .* z
    # Truncation to avoid numerical over/underflow:
    trunc = 8.0 
    z = clamp.(z,-trunc,trunc)
    p = exp.(z)
    p = p ./ (1 .+ p)
    return p
end
