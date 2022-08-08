# using RCall
# import Plots: plot
# using DataFrames
# using Statistics
# using Images

# function plot(results::ExperimentResults, variable::Symbol=:mmd, scope::Symbol=:model; kwargs...)
    
#     df = results.output
#     @assert variable in unique(df.name) "Not a valid variable."

#     gdf = groupby(df, [:generator, :model, :n, :name, :scope])
#     df_plot = combine(gdf, :value => (x -> [(mean(x),mean(x)+std(x),mean(x)-std(x))]) => [:mean, :ymax, :ymin])
#     df_plot = df_plot[(df_plot.scope .== scope),:]
#     df_plot = mapcols(x -> typeof(x) == Vector{Symbol} ? string.(x) : x, df_plot)

#     ncol = length(unique(df_plot.model))
#     @rimport ggplot2 as gg
#     gg.ggplot(df_plot,aes(x=:n, y=:mean, color=:generator)) +
#         gg.geom_line() +
#         gg.facet_wrap(R"name ~ model", scale="free_y", ncol=ncol) 
#     temp_path = joinpath(tempdir(),"plot.png")
#     gg.ggsave(temp_path)
#     img = Images.load(temp_path)

#     return img

# end

# function plot(results::ExperimentResults, n::Int, variable::Symbol=:mmd, scope::Symbol=:model; kwargs...)
    
#     df = results.output
#     @assert variable in unique(df.name) "Not a valid variable."
#     @assert n in unique(df.n) "No results for round `n`."
#     df = df[df.n .== n,:]

#     gdf = groupby(df, [:generator, :model, :name, :scope])
#     df_plot = combine(gdf, :value => (x -> [(mean(x),mean(x)+std(x),mean(x)-std(x))]) => [:mean, :ymax, :ymin])
#     df_plot = df_plot[(df_plot.scope .== scope),:]
#     df_plot = mapcols(x -> typeof(x) == Vector{Symbol} ? string.(x) : x, df_plot)

#     ncol = length(unique(df_plot.model))
#     @rimport ggplot2 as gg
#     gg.ggplot(df_plot) +
#         gg.geom_bar(aes(x=:generator, y=:mean), stat="identity", fill="skyblue", alpha=0.5) +
#         gg.geom_pointrange( aes(x=:generator, y=:mean, ymin=:ymin, ymax=:ymax), colour="orange", alpha=0.9, size=1.3) +
#         gg.facet_wrap(R"name ~ model", scale="free_y", ncol=ncol) 
#     temp_path = joinpath(tempdir(),"plot.png")
#     gg.ggsave(temp_path)
#     img = Images.load(temp_path)

#     return img

# end

