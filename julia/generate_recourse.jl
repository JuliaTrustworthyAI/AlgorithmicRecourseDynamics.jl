struct Recourse
    x_cf::Vector{Float64}
    path::Matrix{Float64}
    valid::Bool
end;

# ------------------------- Wachter et al (2018) -------------------------
function gradient_cost(x_f, x_cf)
    (x_f - x_cf) ./ norm(x_f - x_cf)
end;

function generate_recourse_wachter(x, gradient, classifier, target; α=1, τ=1e-5, λ=0.1, gradient_cost=gradient_cost, T=1000)
    w = coef(classifier)
    D = length(x) # input dimension
    path = reshape(x, 1, D) # storing the path
    # Initialize:
    x_cf = copy(x) # start from factual
    t = 1 # counter
    function convergence_condition(x_cf, gradient, w, target, tol)
        all(gradient(x_cf,w,target) .<= τ)
    end
    converged = convergence_condition(vcat(1,x_cf), gradient, w, target, τ)
    # Recursion:
    while !converged && t < T 
        𝐠_t = gradient(vcat(1,x_cf),w,target)[2:length(w)] # compute gradient
        𝐠_cost_t = gradient_cost(vcat(1,x),vcat(1,x_cf))[2:length(w)] # compute gradient of cost function
        cost = norm(vcat(1,x)-vcat(1,x_cf)) # update cost
        if cost != 0
            x_cf -= α .* (𝐠_t - λ .* 𝐠_cost_t) # counterfactual update
        else
            x_cf -= α .* 𝐠_t
        end
        t += 1 # update number of times feature is changed
        converged = convergence_condition(vcat(1,x_cf), gradient, w, target, τ) # check if converged
        path = vcat(path, reshape(x_cf, 1, D))
    end
    # Validity:
    valid = predict(classifier, vcat(1,x_cf), false)[1] == target * 1.0
    # Output:
    recourse = Recourse(x_cf, path, valid) 
end;

# ------------------------- Upadhyay et al (2021) -------------------------
function gradient_cost(x_f, x_cf)
    (x_f - x_cf) ./ norm(x_f - x_cf)
end;

function generate_recourse_roar(x, gradient, classifier, target; α=1, τ=1e-5, λ=0.25, gradient_cost=gradient_cost, T=1000)
    w = coef(classifier)
    D = length(x) # input dimension
    path = reshape(x, 1, D) # storing the path
    # Initialize:
    x_cf = copy(x) # start from factual
    t = 1 # counter
    function convergence_condition(x_cf, gradient, w, target, tol)
        all(gradient(x_cf,w,target) .<= τ)
    end
    converged = convergence_condition(vcat(1,x_cf), gradient, w, target, τ)
    # Recursion:
    while !converged && t < T 
        𝐠_t = gradient(vcat(1,x_cf),w,target)[2:length(w)] # compute gradient
        𝐠_cost_t = gradient_cost(vcat(1,x),vcat(1,x_cf))[2:length(w)] # compute gradient of cost function
        cost = norm(vcat(1,x)-vcat(1,x_cf)) # update cost
        if cost != 0
            x_cf -= α .* (𝐠_t - λ .* 𝐠_cost_t) # counterfactual update
        else
            x_cf -= α .* 𝐠_t
        end
        t += 1 # update number of times feature is changed
        converged = convergence_condition(vcat(1,x_cf), gradient, w, target, τ) # check if converged
        path = vcat(path, reshape(x_cf, 1, D))
    end
    # Validity:
    valid = predict(classifier, vcat(1,x_cf), false)[1] == target * 1.0
    # Output:
    recourse = Recourse(x_cf, path, valid) 
end;

# ------------------------- Schut et al (2021) -------------------------
function generate_recourse_schut(x,gradient,classifier,target;Γ=0.95,δ=1,n=30,T=100)
    w = coef(classifier)
    D = length(x) # input dimension
    path = reshape(x, 1, D) # storing the path
    # Initialize:
    x_cf = copy(x) # start from factual
    t = 1 # counter
    P = zeros(D) # number of times feature is changed
    converged = posterior_predictive(classifier, vcat(1,x_cf))[1] .> Γ # check if converged
    max_number_changes_reached = all(P.==n)
    # Recursion:
    while !converged && t < T && !max_number_changes_reached
        𝐠_t = gradient(vcat(1,x_cf),w,target)[2:length(w)] # compute gradient
        𝐠_t[P.==n] .= 0 # set gradient to zero, if already changed n times 
        i_t = argmax(abs.(𝐠_t)) # choose most salient feature
        x_cf[i_t] -= δ * sign(𝐠_t[i_t]) # counterfactual update
        P[i_t] += 1 # update 
        t += 1 # update number of times feature is changed
        converged = posterior_predictive(classifier, vcat(1,x_cf))[1] .> Γ # check if converged
        max_number_changes_reached = all(P.==n)
        path = vcat(path, reshape(x_cf, 1, D))
    end
    # Validity:
    valid = predict(classifier, vcat(1,x_cf), false)[1] == target * 1.0
    # Output:
    recourse = Recourse(x_cf, path, valid) 
    return recourse
end;