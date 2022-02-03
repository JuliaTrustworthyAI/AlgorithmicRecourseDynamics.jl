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

using BSON
"""
    save_path(root,path)

Helper function to save `path` output from `run_experiment` to BSON.
"""
function save_path(root,path)
    bson(root * "_path.bson",Dict(i => path[i] for i ∈ 1:length(path)))
end

using BSON
"""
    load_path(root,path)

Helper function to load `path` output.
"""
function load_path(root)
    dict = BSON.load(root * "_path.bson")
    path = [dict[i] for i ∈ 1:length(dict)]
    return path
end

using Plots
"""
    plot_data!(plt,X,y)

Helper function to plut features coloured by label.
"""
function plot_data!(plt,X,y)
    Plots.scatter!(plt, X[y.==1.0,1],X[y.==1.0,2], color=1, clim = (0,1), label="y=1")
    Plots.scatter!(plt, X[y.==0.0,1],X[y.==0.0,2], color=0, clim = (0,1), label="y=0")
end

using Plots, AlgorithmicRecourse
"""
    plot_contour(X,y,𝑴;clegend=true,title="",length_out=100)

Helper function to plot contour of predictive probabilities.
"""
function plot_contour(X,y,𝑴;clegend=true,title="",length_out=100)
    x_range = collect(range(minimum(X[:,1]),stop=maximum(X[:,1]),length=length_out))
    y_range = collect(range(minimum(X[:,2]),stop=maximum(X[:,2]),length=length_out))
    Z = [AlgorithmicRecourse.Models.probs(𝑴,[x, y])[1] for x=x_range, y=y_range]
    plt = contourf(x_range, y_range, Z', color=:viridis, legend=clegend, title=title, linewidth=0)
    plot_data!(plt,X,y)
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