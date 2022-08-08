using RCall
import Plots: plot
using DataFrames
using Statistics
using Images

function plot(results::ExperimentResults, variable::Symbol=:mmd, scope::Symbol=:model; size=3, title=nothing, kwargs...)
    
    df = results.output
    @assert variable in unique(df.name) "Not a valid variable."

    gdf = groupby(df, [:generator, :model, :n, :name, :scope])
    df_plot = combine(gdf, :value => (x -> [(mean(x),mean(x)+std(x),mean(x)-std(x))]) => [:mean, :ymax, :ymin])
    df_plot = df_plot[(df_plot.scope .== scope),:]
    df_plot = mapcols(x -> typeof(x) == Vector{Symbol} ? string.(x) : x, df_plot)

    ncol = length(unique(df_plot.model))
    nrow = length(unique(df_plot.name))
    R"""
    library(ggplot2)
    plt <- ggplot($df_plot,aes(x=n, y=mean, color=generator)) +
        geom_line() +
        facet_wrap(name ~ model, scale="free_y", ncol=$ncol) +
        labs(
            x = "Round",
            y = "Value",
            title = $title
        )
    temp_path <- file.path(tempdir(), "plot.png")
    ggsave(temp_path,width=$ncol * $size,height=$nrow * $size * 0.8)
    """

    img = Images.load(rcopy(R"temp_path"))
    return img

end

function plot(results::ExperimentResults, n::Int, variable::Symbol=:mmd, scope::Symbol=:model; size=3, title=nothing, kwargs...)
    
    df = results.output
    @assert variable in unique(df.name) "Not a valid variable."
    @assert n in unique(df.n) "No results for round `n`."
    df = df[df.n .== n,:]

    gdf = groupby(df, [:generator, :model, :n, :name, :scope])
    df_plot = combine(gdf, :value => (x -> [(mean(x),mean(x)+std(x),mean(x)-std(x))]) => [:mean, :ymax, :ymin])
    df_plot = df_plot[(df_plot.scope .== scope),:]
    df_plot = mapcols(x -> typeof(x) == Vector{Symbol} ? string.(x) : x, df_plot)

    ncol = length(unique(df_plot.model))
    nrow = length(unique(df_plot.name))
    R"""
    library(ggplot2)
    plt <- ggplot($df_plot) +
        geom_bar(aes(x=n, y=mean, fill=generator), stat="identity", alpha=0.5, position="dodge") +
        geom_pointrange( aes(x=n, y=mean, ymin=ymin, ymax=ymax, colour=generator), alpha=0.9, position=position_dodge(width=0.9), size=1.0) +
        facet_wrap(name ~ model, scale="free_y", ncol=$ncol) +
        labs(
            x = "Round",
            y = "Value",
            title = $title
        ) + 
        theme(
            axis.title.x=element_blank(),
            axis.text.x=element_blank(),
            axis.ticks.x=element_blank()
        )
    temp_path <- file.path(tempdir(), "plot.png")
    ggsave(temp_path,width=$ncol * $size,height=$nrow * $size * 0.8) 
    """
    
    img = Images.load(rcopy(R"temp_path"))
    return img

end

