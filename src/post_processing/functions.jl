using DataFrames

"""
    prepare_results(outcome; id_vars=[:t, :μ, :γ])

Helper function that prepares the experimental results for plotting.
"""
function prepare_results(outcome; id_vars=[:t, :μ, :γ])
    validity = DataFrames.groupby(outcome, id_vars) |>
        gdf -> DataFrames.combine(gdf, :pct_valid .=> [mean, std] .=> [:mean, :std])
    validity[!,:ymin] = validity[!,:mean] - validity[!,:std]
    validity[!,:ymax] = validity[!,:mean] + validity[!,:std]

    cost = DataFrames.groupby(outcome, id_vars) |>
        gdf -> DataFrames.combine(gdf, :avg_cost .=> [mean, std] .=> [:mean, :std])
    cost[!,:ymin] = cost[!,:mean] - cost[!,:std]
    cost[!,:ymax] = cost[!,:mean] + cost[!,:std];

    return validity, cost
end

using Gadfly, DataFrames
"""
    plot_results(results::NamedTuple; title="")

Helper function that plots the results from an experient.
"""
function plot_results(results::NamedTuple; title="", id_vars=[:t, :μ, :γ])
    
    set_default_plot_size(700px, 600px)
    validity = DataFrame()
    cost = DataFrame()

    # Prepare for plotting:
    for (k, result) in zip(keys(results), results)
        validityₖ, costₖ = prepare_results(result, id_vars=id_vars)
        insertcols!(validityₖ, :Generator => string(k))
        insertcols!(costₖ, :Generator => string(k))
        validity = vcat(validity, validityₖ)
        cost = vcat(cost, costₖ)
    end

    # Plot validity:
    p_validity = Gadfly.plot(
        validity, 
        xgroup="μ", ygroup="γ", x="t", y="mean", ymin=:ymin, ymax=:ymax, color=:Generator,
        Geom.subplot_grid(Geom.point, Geom.errorbar),
        Guide.title("Validity")
    )

    # Plot cost:
    p_cost = Gadfly.plot(
        cost, 
        xgroup="μ", ygroup="γ", x="t", y="mean", ymin=:ymin, ymax=:ymax, color=:Generator,
        Geom.subplot_grid(Geom.point, Geom.errorbar),
        Guide.title("Cost")
    )

    return p_validity, p_cost

end



using Plots
"""
    plot_path(path;γ,μ,k=1,length_out=500)

Generates and returns an animation for a counterfactual `path` returned by `run_experiment`.
"""
function plot_path(path;Γ::AbstractArray,𝙈::AbstractArray,k=1,length_out=500,p_size=300,aspect=(1.2,1),plot_title="",clegend=false)

    anim = Animation()
    layout_ = (length(Γ),length(𝙈))
    T = sort(unique(map(p -> p.t, path)))

    for t ∈ T
        plts = []
        for γ ∈ Γ
            for μ ∈ 𝙈
                chosen = map(p -> p.γ == γ && p.μ == μ && p.t==t && p.k==k, path)
                path_chosen = path[chosen][1]
                X = path_chosen.X̲'
                y = vec(path_chosen.y̲')
                plt = plot_contour(X,y,path_chosen.𝑴,length_out=length_out,title="γ=" * string(γ) * ", μ=" * string(μ),clegend=clegend)
                plts = vcat(plts, plt)
            end
        end
        plt = Plots.plot(plts...,layout=layout_, size=layout_ .* aspect .* p_size,plot_title=plot_title)
        frame(anim, plt)
    end
    
    return anim
end

# Plot data points:
using Plots
"""
    plot_data!(plt,X,y)

# Examples

```julia-repl
using CounterfactualExplanations
using CounterfactualExplanations.Data: toy_data_linear
using CounterfactualExplanations.Utils: plot_data!
X, y = toy_data_linear(100)
plt = plot()
plot_data!(plt, hcat(X...)', y)
```

"""
function plot_data!(plt,X,y)
    Plots.scatter!(plt, X[:,1],X[:,2],group=Int.(y),color=Int.(y))
end

# Plot contour of posterior predictive:
using Plots, CounterfactualExplanations.Models
"""
    plot_contour(X,y,M;colorbar=true,title="",length_out=50,type=:laplace,zoom=0,xlim=nothing,ylim=nothing)

Generates a contour plot for the posterior predictive surface.  

# Examples

```julia-repl
using CounterfactualExplanations
using CounterfactualExplanations.Data: toy_data_linear
using CounterfactualExplanations.Utils: plot_contour
X, y = toy_data_linear(100)
X = hcat(X...)'
β = [1,1]
M =(β=β,)
predict(M, X) = σ.(M.β' * X)
plot_contour(X, y, M)
```

"""
function plot_contour(X,y,M;colorbar=true,title="",length_out=50,zoom=-1,xlim=nothing,ylim=nothing,linewidth=0.1)
    
    # Surface range:
    if isnothing(xlim)
        xlim = (minimum(X[:,1]),maximum(X[:,1])).+(zoom,-zoom)
    else
        xlim = xlim .+ (zoom,-zoom)
    end
    if isnothing(ylim)
        ylim = (minimum(X[:,2]),maximum(X[:,2])).+(zoom,-zoom)
    else
        ylim = ylim .+ (zoom,-zoom)
    end
    x_range = collect(range(xlim[1],stop=xlim[2],length=length_out))
    y_range = collect(range(ylim[1],stop=ylim[2],length=length_out))
    Z = [Models.probs(M,[x, y])[1] for x=x_range, y=y_range]

    # Plot:
    plt = contourf(
        x_range, y_range, Z'; 
        colorbar=colorbar, title=title, linewidth=linewidth,
        xlim=xlim,
        ylim=ylim
    )
    plot_data!(plt,X,y)

end

# Plot contour of posterior predictive:
using Plots, CounterfactualExplanations.Models
"""
    plot_contour_multi(X,y,M;colorbar=true,title="",length_out=50,type=:laplace,zoom=0,xlim=nothing,ylim=nothing)

Generates a contour plot for the posterior predictive surface.  

# Examples

```julia-repl
using BayesLaplace, Plots
import BayesLaplace: predict
using NNlib: σ
X, y = toy_data_linear(100)
X = hcat(X...)'
β = [1,1]
M =(β=β,)
predict(M, X) = σ.(M.β' * X)
plot_contour(X, y, M)
```

"""
function plot_contour_multi(X,y,M;
    target::Union{Nothing,Number}=nothing,title="",colorbar=true, length_out=50,zoom=-1,xlim=nothing,ylim=nothing,linewidth=0.1)
    
    # Surface range:
    if isnothing(xlim)
        xlim = (minimum(X[:,1]),maximum(X[:,1])).+(zoom,-zoom)
    else
        xlim = xlim .+ (zoom,-zoom)
    end
    if isnothing(ylim)
        ylim = (minimum(X[:,2]),maximum(X[:,2])).+(zoom,-zoom)
    else
        ylim = ylim .+ (zoom,-zoom)
    end
    x_range = collect(range(xlim[1],stop=xlim[2],length=length_out))
    y_range = collect(range(ylim[1],stop=ylim[2],length=length_out))
    Z = reduce(hcat, [Models.probs(M,[x, y]) for x=x_range, y=y_range])

    # Plot:
    if isnothing(target)
        # Plot all contours as lines:
        out_dim = size(Z)[1]
        p_list = []
        for d in 1:out_dim
            plt = contourf(
                x_range, y_range, Z[d,:]; 
                colorbar=colorbar, title="p(y=$(string(d))|X)", linewidth=linewidth,
                xlim=xlim,
                ylim=ylim
            )
            plot_data!(plt,X,y)
            p_list = vcat(p_list..., plt)
        end
        plt = plot(p_list..., layout=Int((out_dim)), plot_title=title)
    else
        # Print contour fill of target class:
        plt = contourf(
            x_range, y_range, Z[Int(target),:]; 
            colorbar=colorbar, title=title, linewidth=linewidth,
            xlim=xlim,
            ylim=ylim
        )
        plot_data!(plt,X,y)
    end

    return plt
    
end
