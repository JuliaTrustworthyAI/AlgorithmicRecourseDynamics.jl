# ------------------------- RECOURSE constructor -------------------------
mutable struct Recourse
    x_cf::Vector{Float64}
    y_cf::Float64
    path::Matrix{Float64}
    target::Float64
    valid::Bool
    cost::Float64
    factual::Vector{Float64}
end;

# --------------- Outer constructor methods: 
function valid(self::Recourse; classifier=nothing)
    if isnothing(classifier)
        valid = self.valid
    else 
        valid = predict(classifier, vcat(1, self.x_cf); proba=false)[1] == self.target
    end
    return valid
end

function cost(self::Recourse; cost_fun=nothing, cost_fun_kargs)
    if isnothing(cost_fun)
        cost = self.cost
    else 
        cost = cost_fun(self.factual .- self.x_cf; cost_fun_kargs...)
        self.cost = cost
    end
    return cost
end

# ------------------------- RECOUSE generators -------------------------

# --------------- Wachter et al (2018): 
function gradient_cost(x_f, x_cf)
    (x_cf .- x_f) ./ norm(x_cf .- x_f)
end;

function generate_recourse_wachter(x, gradient, classifier, target; T=1000, immutable_=[], a=1, τ=1e-5, λ=0.1, gradient_cost=gradient_cost)
    # Setup:
    w = coef(classifier)
    constant_needed = length(w) > length(x) # is adjustment for constant needed?
    if (constant_needed)
        x = vcat(1, x)
        immutable_ = vcat(1, immutable_ .+ 1) # adjust mask for immutable features
    end
    x_cf = copy(x) # start from factual
    D = length(x_cf)
    path = reshape(x, 1, length(x)) # storing the path
    function convergence_condition(x_cf, gradient, w, target, tol)
        all(gradient(x_cf,w,target) .<= τ)
    end
    
    # Initialize:
    t = 1 # counter
    converged = convergence_condition(x_cf, gradient, w, target, τ)
    
    # Recursion:
    while !converged && t < T 
        g_t = gradient(x_cf,w,target) # compute gradient
        g_t[immutable_] .= 0 # set gradient of immutable features to zero
        g_cost_t = gradient_cost(x,x_cf) # compute gradient of cost function
        g_cost_t[immutable_] .= 0 # set gradient of immutable features to zero
        cost = norm(x_cf-x) # update cost
        if cost != 0
            x_cf -= (a .* (g_t + λ .* g_cost_t)) # counterfactual update
        else
            x_cf -= (a .* g_t)
        end
        t += 1 # update number of times feature is changed
        converged = convergence_condition(x_cf, gradient, w, target, τ) # check if converged
        path = vcat(path, reshape(x_cf, 1, D))
    end
    
    # Output:
    new_label = predict(classifier, x_cf; proba=false)[1]
    valid = new_label == target * 1.0
    cost = norm(x.-x_cf)
    if (constant_needed)
        path = path[:,2:end]
        x_cf = x_cf[2:end]
        x = x[2:end]
    end
    recourse = Recourse(x_cf, new_label, path, target, valid, cost, x) 
    
    return recourse
end;

# --------------- Upadhyay et al (2021) 
function generate_recourse_roar(x, gradient, classifier, target; T=1000, immutable_=[], a=1, τ=1e-5, λ=0.1, gradient_cost=gradient_cost)
    # Setup:
    w = coef(classifier)
    constant_needed = length(w) > length(x) # is adjustment for constant needed?
    if (constant_needed)
        x = vcat(1, x)
        immutable_ = vcat(1, immutable_ .+ 1) # adjust mask for immutable features
    end
    x_cf = copy(x) # start from factual
    D = length(x_cf)
    path = reshape(x, 1, length(x)) # storing the path
    function convergence_condition(x_cf, gradient, w, target, tol)
        all(gradient(x_cf,w,target) .<= τ)
    end
    
    # Initialize:
    t = 1 # counter
    converged = convergence_condition(x_cf, gradient, w, target, τ)
    
    # Recursion:
    while !converged && t < T 
        g_t = gradient(x_cf,w,target) # compute gradient
        g_t[immutable_] .= 0 # set gradient of immutable features to zero
        g_cost_t = gradient_cost(x,x_cf) # compute gradient of cost function
        g_cost_t[immutable_] .= 0 # set gradient of immutable features to zero
        cost = norm(x_cf-x) # update cost
        if cost != 0
            x_cf -= (a .* (g_t + λ .* g_cost_t)) # counterfactual update
        else
            x_cf -= (a .* g_t)
        end
        t += 1 # update number of times feature is changed
        converged = convergence_condition(x_cf, gradient, w, target, τ) # check if converged
        path = vcat(path, reshape(x_cf, 1, D))
    end
    
    # Output:
    new_label = predict(classifier, x_cf; proba=false)[1]
    valid = new_label == target * 1.0
    cost = norm(x.-x_cf)
    if (constant_needed)
        path = path[:,2:end]
        x_cf = x_cf[2:end]
        x = x[2:end]
    end
    recourse = Recourse(x_cf, new_label, path, target, valid, cost, x) 
    
    return recourse
end;

# --------------- Schut et al (2021) 
function generate_recourse_schut(x,gradient,classifier,target;T=1000,immutable_=[],Γ=0.95,δ=1,n=nothing)
    # Setup:
    w = coef(classifier)
    constant_needed = length(w) > length(x) # is adjustment for constant needed?
    if (constant_needed)
        x = vcat(1, x)
        immutable_ = vcat(1, immutable_ .+ 1) # adjust mask for immutable features
    end
    x_cf = copy(x) # start from factual
    D = length(x_cf)
    D_mutable = length(setdiff(1:D, immutable_))
    path = reshape(x, 1, length(x)) # storing the path
    if isnothing(n)
        n = ceil(T/D_mutable)
    end
    
    # Intialize:
    t = 1 # counter
    P = zeros(D) # number of times feature is changed
    converged = posterior_predictive(classifier, x_cf)[1] .> Γ # check if converged
    max_number_changes_reached = all(P.==n)
    
    # Recursion:
    while !converged && t < T && !max_number_changes_reached
        g_t = gradient(x_cf,w,target) # compute gradient
        g_t[P.==n] .= 0 # set gradient to zero, if already changed n times 
        g_t[immutable_] .= 0 # set gradient of immutable features to zero
        i_t = argmax(abs.(g_t)) # choose most salient feature
        x_cf[i_t] -= δ * sign(g_t[i_t]) # counterfactual update
        P[i_t] += 1 # update 
        t += 1 # update number of times feature is changed
        converged = posterior_predictive(classifier, x_cf)[1] .> Γ # check if converged
        max_number_changes_reached = all(P.==n)
        path = vcat(path, reshape(x_cf, 1, D))
    end
    
    # Output:
    new_label = predict(classifier, x_cf; proba=false)[1]
    valid = new_label == target * 1.0
    cost = norm(x.-x_cf)
    if (constant_needed)
        path = path[:,2:end]
        x_cf = x_cf[2:end]
        x = x[2:end]
    end
    recourse = Recourse(x_cf, new_label, path, target, valid, cost, x) 
    
    return recourse
end;